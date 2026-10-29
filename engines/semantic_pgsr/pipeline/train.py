import os


from argparse import ArgumentParser, Namespace
from pathlib import Path
import random
import math
import json
import numpy as np
import torch
from arguments import ModelParams, PipelineParams, OptimizationParams
from pipeline.model import GaussianModel
from pipeline.active import add_compact_stats_many, ActiveSetCache, CompactModel, active_adam_step_many, add_compact_stats
from pipeline.scene import Scene
from pipeline.sampling import CameraSampler
from pipeline.masks import mask_roi
from pipeline.rendering import render_many
from pipeline.losses import per_view_loss, pair_loss, local_mask
from pipeline.supervision import prepare_supervision, supervision_for, fallback_supervision
from pipeline.guard import load_guard_targets, union_roi, guard_rgb_loss, guard_for
from pipeline.step_retry import backward_with_cache_retry
from scene.app_model import AppModel


USE_TIGHT_NEIGHBOR_ROI = False


def make_parser():
    parser = ArgumentParser(description='SemLite-GS instance refinement with frozen-context protection.')
    model, optimization, pipeline = ModelParams(parser), OptimizationParams(parser), PipelineParams(parser)
    parser.add_argument('--camera_batch_size', type=int, default=4)
    parser.add_argument('--benchmark_timing', action='store_true')
    parser.add_argument('--benchmark_barrier_dir', default='')
    parser.add_argument('--benchmark_worker_id', default='0')
    parser.add_argument('--multi_instance_batch', action='store_true',
                        help='Sample different target instances in one native camera batch')
    parser.add_argument('--compact_active_gaussians', action='store_true')
    parser.add_argument('--compact_stats_mode', choices=['fused','reference'], default='fused',
                        help='Batched CUDA densification statistics, or the original per-view reference')
    parser.add_argument('--render_inputs_mode', choices=['reference','axis','fused'], default='fused',
                        help='Native 3D render inputs: original, fused axis, or fused axis and camera maps')
    parser.add_argument('--raster_context_bbox_padding', type=float, default=-1.)
    parser.add_argument('--image_cache_size', type=int, default=8)
    parser.add_argument('--camera_io_workers', type=int, default=0)
    parser.add_argument('--guard_io_workers', type=int, default=0)
    parser.add_argument('--fast_init_ply', action='store_true', help='Skip temporary COLMAP Gaussians; preserve RNG consumption')
    parser.add_argument('--fast_point_budget', action='store_true', help='Aggregate independent-owner budget counts')
    parser.add_argument('--supervision_cache', choices=['gpu','cpu','none'], default='gpu')
    parser.add_argument('--min_instance_mask_area',type=int,default=200)
    parser.add_argument('--skip_initial_mask_prune',action='store_true')
    parser.add_argument('--skip_occluded_frozen_prune',action='store_true')
    parser.add_argument('--initial_prune_mask_dilation',type=int,default=0)
    parser.add_argument('--occluded_frozen_depth_margin',type=float,default=1.)
    parser.add_argument('--occluded_frozen_alpha_threshold',type=float,default=.5)
    parser.add_argument('--occluded_frozen_occluded_ratio',type=float,default=.8)
    parser.add_argument('--occluded_frozen_min_relevant_views',type=int,default=1)
    parser.add_argument('--occluded_frozen_render_batch_size',type=int,default=2)
    parser.add_argument('--global_freeze_xy_padding',type=float,default=0.)
    parser.add_argument('--global_freeze_z_padding',type=float,default=0.)
    parser.add_argument('--global_freeze_trim_quantile',type=float,default=.02)
    parser.add_argument('--prune_final_to_building_prisms',action='store_true')
    parser.add_argument('--final_prune_xy_padding',type=float,default=.5)
    parser.add_argument('--final_prune_z_padding',type=float,default=.5)
    parser.add_argument('--final_prune_trim_quantile',type=float,default=.005)
    parser.add_argument('--allow_full_union_context', action='store_true', help='Explicit reference mode: retain the complete frozen union in each mixed view')
    parser.add_argument('--instance_context_manifest', default='', help='Validated per-instance candidate PLY and guard manifest')
    parser.add_argument('--instance_context_mode', choices=['independent','joint'], default='independent',
                        help='Independent targets use immutable initial context; joint retains evolving other owners')
    parser.add_argument('--reference_losses', action='store_true', help='Use serial PyTorch losses for numerical/performance comparison')
    parser.add_argument('--instance_masks', default='')
    parser.add_argument('--instance_ids', default='')
    parser.add_argument('--mask_background_id',type=int,default=255)
    parser.add_argument('--building_ply', default='')
    parser.add_argument('--building_transform', default='')
    parser.add_argument('--building_colors', default='')
    parser.add_argument('--building_values', default='')
    parser.add_argument('--init_ply', default='')
    parser.add_argument('--init_ply_translation', nargs=3, type=float, default=None,
                        metavar=('DX', 'DY', 'DZ'),
                        help='World-space translation from FastGS PLY coordinates to COLMAP scene coordinates')
    parser.add_argument('--freeze_global_outside_instances', action='store_true')
    parser.add_argument('--frozen_context_guard_dir', default='')
    parser.add_argument('--frozen_context_guard_weight', type=float, default=0.3)
    parser.add_argument('--frozen_context_guard_start_iter', type=int, default=1)
    parser.add_argument('--frozen_context_guard_end_iter', type=int, default=0)
    parser.add_argument('--max_points_per_instance', type=int, default=0)
    parser.add_argument('--plane_evidence', action='store_true')
    parser.add_argument('--plane_render_batch_size', type=int, default=8)
    parser.add_argument('--plane_constraint_from_iter', type=int, default=20000)
    parser.add_argument('--plane_hard_support_multiplier', type=float, default=3.0)
    parser.add_argument('--plane_export_interval', type=int, default=0)
    parser.add_argument('--save_iterations', nargs='+', type=int, default=[7000,30000])
    parser.add_argument('--checkpoint_iterations', nargs='+', type=int, default=[])
    parser.add_argument('--start_checkpoint', default='')
    parser.add_argument('--seed', type=int, default=22)
    return parser, model, optimization, pipeline


def training(dataset, opt, pipe, args):
    from pipeline.benchmark_timing import TrainingTimer
    timer = TrainingTimer(dataset.model_path, getattr(args,'benchmark_timing',False),
                          getattr(args,'benchmark_barrier_dir',''),
                          getattr(args,'benchmark_worker_id','0'))
    from pipeline import render_inputs
    render_inputs.configure(getattr(args, 'render_inputs_mode', 'fused'))
    if args.camera_batch_size < 1 or args.plane_render_batch_size < 1 or args.image_cache_size < 1:
        raise ValueError('camera_batch_size, plane_render_batch_size and image_cache_size must be positive')
    if args.multi_instance_batch and not getattr(args,'instance_context_manifest','') and not getattr(args,'allow_full_union_context',False):
        raise ValueError('Mixed training requires --instance_context_manifest with validated owner subsets; --allow_full_union_context is reference-only')
    if args.multi_instance_batch and (not args.compact_active_gaussians or not args.freeze_global_outside_instances):
        raise ValueError('Mixed-instance batches require compact_active_gaussians and freeze_global_outside_instances')
    if args.frozen_context_guard_dir and not args.freeze_global_outside_instances:
        raise ValueError('Frozen-context protection requires freeze_global_outside_instances')
    if args.frozen_context_guard_weight < 0 or args.frozen_context_guard_start_iter < 1 or args.frozen_context_guard_end_iter < 0:
        raise ValueError('Invalid frozen-context guard weight or iteration window')
    if args.frozen_context_guard_end_iter and args.frozen_context_guard_end_iter <= args.frozen_context_guard_start_iter:
        raise ValueError('Guard end iteration must exceed start iteration')
    if args.init_ply_translation is not None and (not args.init_ply or
            not all(math.isfinite(value) for value in args.init_ply_translation)):
        raise ValueError('init_ply_translation requires init_ply and three finite coordinates')
    if args.instance_context_manifest and args.instance_context_mode == 'independent':
        if not args.compact_active_gaussians or not args.freeze_global_outside_instances:
            raise ValueError('Independent contexts require compact_active_gaussians and freeze_global_outside_instances')
        if not args.skip_initial_mask_prune or not args.skip_occluded_frozen_prune:
            raise ValueError('Validated independent contexts require skip_initial_mask_prune and skip_occluded_frozen_prune; do not re-prune validated occluders')
        if opt.exposure_compensation:
            raise ValueError('Independent context currently requires exposure_compensation=False (shared exposure parameters couple owners)')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    timer.track_gpu()
    Path(dataset.model_path).mkdir(parents=True, exist_ok=True)
    (Path(dataset.model_path)/'cfg_args').write_text(str(Namespace(**vars(dataset))))
    model = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, model, args)
    if args.instance_context_manifest and args.instance_context_mode == 'independent':
        expected = len(scene.instance_ids) if args.multi_instance_batch else 1
        if args.camera_batch_size != expected or (not args.multi_instance_batch and len(scene.instance_ids) != 1):
            raise ValueError('Independent training requires one reference view per owner per step: use multi_instance_batch with camera_batch_size equal to owner count, or one owner with batch size 1')
    model.training_setup(opt)
    cameras = scene.getTrainCameras()
    sampler = CameraSampler(cameras, scene.instance_ids, args.seed)
    app = AppModel(num_images=max(1600,max(c.uid for c in cameras)+1)).cuda().train()
    first = 0
    if args.start_checkpoint:
        checkpoint = torch.load(args.start_checkpoint, map_location='cuda', weights_only=False)
        model.restore(checkpoint['model'], opt)
        sampler.restore(checkpoint['sampler'])
        first = checkpoint['iteration']
        app.load_state_dict(checkpoint['app'])
        app.optimizer.load_state_dict(checkpoint['app_optimizer'])
        random.setstate(checkpoint['random'])
        np.random.set_state(checkpoint['numpy'])
        torch.set_rng_state(checkpoint['torch'].cpu())
        torch.cuda.set_rng_state_all([s.cpu() for s in checkpoint['cuda']])


        del checkpoint
    active_cache = ActiveSetCache(getattr(args, 'raster_context_bbox_padding', -1.)) if getattr(args, 'compact_active_gaussians', False) else None
    if active_cache is not None and not model.freeze_global:
        raise ValueError('Compact instance training requires frozen global context')
    background = torch.tensor([1,1,1] if dataset.white_background else [0,0,0], device='cuda', dtype=torch.float32)
    coverage = {i: sum(any(j in sampler.eligible[i] for j in cameras[k].nearest_id if j != k)
                       for k in sampler.eligible[i]) for i in scene.instance_ids}
    print('Multi-view neighbor coverage:', {i: f'{coverage[i]}/{len(sampler.eligible[i])}' for i in coverage})
    if opt.iterations > opt.multi_view_weight_from_iter and not any(coverage.values()) and not opt.use_virtul_cam:
        raise ValueError('No eligible multiview camera pairs. Set multi_view_max_dis/min_dis in scene units and max_angle appropriately.')

    cache_cameras = [cameras[i] for i in sorted(set(j for ids in sampler.eligible.values() for j in ids))]
    if args.frozen_context_guard_dir and not getattr(args,'instance_context_manifest',''):
        load_guard_targets(cache_cameras, args.frozen_context_guard_dir, args.guard_io_workers)
    if getattr(args, 'instance_context_manifest', ''):
        from pipeline.context import install_instance_context
        install_instance_context(model, cameras, sampler, args.instance_context_manifest, resume=bool(args.start_checkpoint), mode=args.instance_context_mode, guard_io_workers=args.guard_io_workers)
    if not args.start_checkpoint:
        from pipeline.initialization import initial_mask_keep, occluded_frozen_keep
        initial_indices = (model.render_order.clone() if model.context_mode == "independent" else
                           torch.arange(len(model.get_xyz), device=model.get_xyz.device))
        initialization = {'source_points': model.next_render_order if model.context_mode == 'independent' else len(initial_indices),
                          'context_mode': model.context_mode,
                          'shared_snapshot_points': len(initial_indices)-model.next_render_order if model.context_mode == 'independent' else 0,
                          'eligible_cameras': len(cache_cameras),
                          'context_bbox_padding': args.raster_context_bbox_padding}
        if args.instance_masks and not args.skip_initial_mask_prune:
            print('Initial mask projection: selecting relevant scene points', flush=True)
            keep, report = initial_mask_keep(model.get_xyz, cache_cameras, scene.instance_ids,
                                             args.initial_prune_mask_dilation)
            model.prune_initial_points(~keep)
            initial_indices = initial_indices[keep]
            initialization['mask_projection'] = report
            print('Initial mask projection:', report, flush=True)
        if not model.trainable_mask.any():
            raise ValueError('Initial mask filtering left no trainable point inside selected building prisms')

        initial_indices = initial_indices[model.enforce_initial_budgets()]
        if args.instance_masks and model.freeze_global and not args.skip_occluded_frozen_prune:
            print('Initial background occlusion: retaining any exposed context', flush=True)
            keep, report = occluded_frozen_keep(model, cache_cameras, scene.instance_ids, pipe, background,
                depth_margin=args.occluded_frozen_depth_margin,
                alpha_threshold=args.occluded_frozen_alpha_threshold,
                occluded_ratio=args.occluded_frozen_occluded_ratio,
                min_relevant_views=args.occluded_frozen_min_relevant_views,
                render_batch_size=args.occluded_frozen_render_batch_size)
            model.prune_initial_points(~keep)
            initial_indices = initial_indices[keep]
            initialization['occluded_frozen'] = report
            print('Initial background occlusion:', report, flush=True)

        torch.save(initial_indices.cpu(), Path(scene.model_path)/'initial_source_indices.pt')
        initialization.update(total_points=len(model.get_xyz),
            trainable_points=int(model.trainable_mask.sum()), frozen_points=int((~model.trainable_mask).sum()),
            source_indices_stage='iteration_0')
        (Path(scene.model_path)/'initialization_report.json').write_text(json.dumps(initialization,indent=2))
        scene.save(0)
        print('Training population:', initialization, flush=True)
    cache_mode = getattr(args,'supervision_cache','gpu')
    supervision_status = (prepare_supervision(cache_cameras,scene.instance_ids,
                          device='cuda' if cache_mode == 'gpu' else 'cpu',
                          guarded=bool(args.frozen_context_guard_dir))
                          if cache_mode != 'none' else None)
    has_planes = bool(model.plane_assigned.any())

    def selection(camera, instance, guard_active=False, tight=False):
        if guard_active:
            entry = supervision_for(camera, instance)
            if entry is not None:
                return entry.mask, entry.roi
            mask = camera.instance_mask == instance
            return mask, union_roi(mask_roi(mask),
                                   getattr(guard_for(camera, instance), 'roi', None))
        entry = supervision_for(camera,instance)
        if entry is not None:
            roi = entry.roi
            if tight and entry.mask_roi is not None:

                x,y,w,h=entry.mask_roi;rx,ry,rw,rh=roi
                left,top=max(rx,x-1),max(ry,y-1)
                roi=(left,top,min(rx+rw,x+w+1)-left,min(ry+rh,y+h+1)-top)
            return entry.mask, roi
        mask = None if camera.instance_mask is None else camera.instance_mask == instance
        return mask, None if mask is None else mask_roi(mask)

    def release_supervision():
        nonlocal supervision_status
        supervision_status = fallback_supervision(cache_cameras)

    def clear_failed_gradients():
        model.optimizer.zero_grad(set_to_none=True)
        app.optimizer.zero_grad(set_to_none=True)

    timer.ready()
    for iteration in range(first+1,opt.iterations+1):
        model.update_learning_rate(iteration)
        if iteration % 1000 == 0:
            model.oneupSHdegree()
        model.use_app = iteration > 1000 and opt.exposure_compensation
        if args.multi_instance_batch:
            chosen = sampler.sample_pairs(args.camera_batch_size)
            selected_instances = [owner for owner, _ in chosen]
            indices = [index for _, index in chosen]
        else:
            instance, indices = sampler.sample(args.camera_batch_size)
            selected_instances = [instance] * len(indices)
        batch = [cameras[k] for k in indices]
        all_cameras = list(batch)
        all_instances = list(selected_instances)
        pairs = []
        if iteration > opt.multi_view_weight_from_iter:
            for k,c,instance in zip(indices,batch,selected_instances):
                choices = [j for j in c.nearest_id if j != k and j in sampler.eligible[instance]]
                virtual = opt.use_virtul_cam and (random.random() < opt.virtul_cam_prob or not choices)
                if virtual:

                    from pipeline.cameras import gen_virtul_cam
                    near = gen_virtul_cam(c, dataset.multi_view_max_dis, dataset.multi_view_max_angle)
                    near.instance_mask = None
                elif choices:
                    near = cameras[random.choice(choices)]
                else:
                    pairs.append(None)
                    continue
                pairs.append((len(all_cameras),virtual))
                all_cameras.append(near)
                all_instances.append(instance)
        else:
            pairs = [None] * len(batch)
        bg = torch.rand(3,device='cuda') if opt.random_background else background
        guard_active = bool(args.frozen_context_guard_dir and args.frozen_context_guard_weight and
                            iteration >= args.frozen_context_guard_start_iter and
                            (not args.frozen_context_guard_end_iter or iteration < args.frozen_context_guard_end_iter))
        batch_instance = (selected_instances[0] if len(set(selected_instances)) == 1
                          else tuple(selected_instances))
        observer = getattr(args, '_iteration_observer', None)
        if observer is not None:
            observer('before_forward', iteration, model, {'indices':indices,'near_ids':[c.uid for c in all_cameras],
                'instance':batch_instance,'sampler':sampler.capture(),'python_rng':random.getstate(),
                'numpy_rng':tuple(v.tolist() if isinstance(v,np.ndarray) else v for v in np.random.get_state()),
                'torch_rng':torch.get_rng_state()})
        def compute_step():


            selections = [(None,None) if getattr(c,'is_virtual',False) else
                          selection(c,owner,guard_active and index < len(batch),
                                    tight=USE_TIGHT_NEIGHBOR_ROI and index >= len(batch) and iteration >= opt.densify_until_iter)
                          for index,(c,owner) in enumerate(zip(all_cameras,all_instances))]
            all_masks, all_rois = zip(*selections)
            if active_cache is not None and os.environ.get('PGSR_NATIVE_COMPACTS','0') == '1':
                from pipeline.native_compacts import prepare
                compact_models = prepare(model, {owner: active_cache.select(model, owner)
                                                 for owner in set(selected_instances)})
            else:
                compact_models = ({owner: CompactModel(model,active_cache.select(model,owner),owner)
                                   for owner in set(selected_instances)} if active_cache is not None else {})
            render_models = [compact_models[owner] if active_cache is not None else model
                             for owner in all_instances]
            packages = render_many(all_cameras,model,pipe,bg,rois=all_rois,models=render_models,app_model=app,
                return_plane=iteration>opt.single_view_weight_from_iter or iteration>opt.multi_view_weight_from_iter,
                return_depth_normal=iteration>opt.single_view_weight_from_iter,
                depth_normal_mask=[i < len(batch) for i in range(len(all_cameras))])
            loss = (next(iter(compact_models.values())).get_xyz.sum()*0
                    if active_cache is not None else model.get_xyz.sum()*0)
            for index, owner in enumerate(selected_instances):
                packages[index]['instance_id'] = owner
            if not getattr(args, 'reference_losses', False):
                from pipeline.loss_batch import per_view_losses
                loss = loss + per_view_losses(batch,packages[:len(batch)],render_models[:len(batch)],
                    opt,iteration,all_masks[:len(batch)],
                    args.frozen_context_guard_weight if guard_active else 0.).sum()
            batch_pairs = []
            for index,(camera,mask,pair,owner) in enumerate(zip(batch,all_masks,pairs,selected_instances)):
                if getattr(args, 'reference_losses', False):
                    loss = loss + per_view_loss(camera,packages[index],render_models[index],opt,iteration,mask)
                    if guard_active:
                        loss = loss + args.frozen_context_guard_weight*guard_rgb_loss(packages[index],camera)
                if pair is not None:
                    near_index, virtual = pair
                    pkg = packages[near_index]
                    near_mask = all_masks[near_index]
                    if near_mask is not None:
                        pkg['loss_mask'] = local_mask(near_mask,pkg)
                    if virtual:
                        from copy import copy
                        pair_opt = copy(opt)
                        pair_opt.multi_view_ncc_weight = 0.0
                    else:
                        pair_opt = opt
                    if getattr(args, 'reference_losses', False):
                        loss = loss + pair_loss(camera,all_cameras[near_index],packages[index],pkg,model,pair_opt,mask)
                    else:
                        batch_pairs.append((camera,all_cameras[near_index],packages[index],pkg,mask,virtual))
            if batch_pairs:
                from pipeline.pair_batch import pair_losses
                cs,ns,ps,nps,ms,vs = zip(*batch_pairs)
                loss = loss + pair_losses(cs,ns,ps,nps,opt,ms,vs).sum()


            if model.context_mode != 'independent':
                loss = loss / len(batch)

            loss_value = float(loss.detach() / (len(batch) if model.context_mode == 'independent' else 1))
            if not math.isfinite(loss_value):
                raise FloatingPointError(f'Nonfinite training loss at iteration {iteration}')
            loss.backward()
            return loss,loss_value,packages,compact_models,render_models

        loss,loss_value,packages,compact_models,render_models = backward_with_cache_retry(
            compute_step,release_supervision,clear_failed_gradients,
            supervision_status is not None and supervision_status.device.startswith('cuda'))
        if observer is not None:
            observer('after_backward', iteration, model, {'loss':loss_value,
                    'active_points':sum(len(c.get_xyz) for c in compact_models.values()) if active_cache is not None else len(model.get_xyz),
                    'trainable_points':(sum(len(c.optimizer_indices) for c in compact_models.values()) if active_cache is not None
                                        else int(model.trainable_mask.sum()))})
        with torch.no_grad():

            if iteration < opt.densify_until_iter and active_cache is not None:
                add_compact_stats_many(model,packages,render_models,
                    1 if model.context_mode == 'independent' else len(batch),
                    fused=getattr(args,'compact_stats_mode','fused') == 'fused')
            if iteration < opt.densify_until_iter and active_cache is None:
                for pkg,render_model in zip(packages,render_models):
                    visible = pkg['visibility_filter'] & model.trainable_mask
                    observed = visible & (pkg['out_observe'].reshape(-1)>0)
                    model.max_radii2D[observed] = torch.maximum(model.max_radii2D[observed],pkg['radii'][observed])
                    signed,absolute = pkg['viewspace_points'],pkg['viewspace_points_abs']
                    if signed.grad is not None and absolute.grad is not None:

                        signed.grad.mul_(len(batch))
                        absolute.grad.mul_(len(batch))
                        model.add_densification_stats(signed,absolute,visible)
            if active_cache is None:
                model.optimizer_step()
            else:
                active_adam_step_many(model,compact_models.values())
            if has_planes:
                from pipeline.planes.constraints import enforce_plane_constraints
                enforce_plane_constraints(model)
            app.optimizer.step()
            app.optimizer.zero_grad(set_to_none=True)
            if iteration < opt.densify_until_iter:
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    model.densify_and_prune(opt.densify_grad_threshold,opt.densify_abs_grad_threshold,
                        opt.opacity_cull_threshold,scene.cameras_extent,20 if iteration>opt.opacity_reset_interval else None)
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    model.reset_opacity()
                if opt.use_multi_view_trim and iteration % 1000 == 0:


                    owners = scene.instance_ids if active_cache is not None else [None]
                    remove = torch.zeros(len(model.get_xyz),device='cuda',dtype=torch.bool)
                    for owner in owners:
                        if owner is not None and not (model.instance_ids == owner).any():
                            continue
                        trim_model = (CompactModel(model,active_cache.select(model,owner),owner)
                                      if active_cache is not None else model)
                        counts = torch.zeros(len(trim_model.get_xyz),device='cuda',dtype=torch.int32)
                        for start in range(0,len(cameras),args.camera_batch_size):
                            for pkg in render_many(cameras[start:start+args.camera_batch_size],trim_model,pipe,bg,app_model=app,return_plane=False,return_depth_normal=False):
                                counts += (pkg['out_observe'].reshape(-1)>0).int()
                        if active_cache is None:
                            remove |= counts < 2
                        else:
                            remove[trim_model.global_indices] |= (counts < 2) & trim_model.trainable_mask
                        del trim_model
                    model.prune_points(remove)
            if args.plane_evidence and iteration == args.plane_constraint_from_iter:
                from pipeline.planes.export_evidence import export_dynamic_evidence
                from pipeline.planes.constraints import assign_plane_constraints, enforce_plane_constraints
                prior = export_dynamic_evidence(model,cameras,scene.instance_ids,pipe,background,args,
                    str(Path(scene.model_path)/'dynamic_plane_prior'/f'iteration_{iteration}'))
                assign_plane_constraints(model,prior,args)
                has_planes = bool(model.plane_assigned.any())
                enforce_plane_constraints(model)
            if iteration == opt.iterations and getattr(args,'prune_final_to_building_prisms',False) and scene.final_prism_models:
                from pipeline.planes.building_prisms import points_in_prisms_numpy
                remove = torch.zeros(len(model.get_xyz),dtype=torch.bool,device='cuda')
                xyz = model.get_xyz.detach().cpu().numpy()
                for owner, prism in scene.final_prism_models.items():
                    inside = torch.as_tensor(points_in_prisms_numpy(xyz,prism),device='cuda')
                    remove |= (model.instance_ids == owner) & ~inside
                model.prune_points(remove)
            if iteration % 10 == 0 or iteration == opt.iterations:
                print(f'iteration={iteration} loss={loss_value:.6f} points={len(model.get_xyz)} instances={selected_instances}')
            if iteration in args.save_iterations or iteration == opt.iterations:
                with timer.saving():
                    scene.save(iteration)
                    app.save_weights(scene.model_path,iteration)
            if iteration in args.checkpoint_iterations or iteration == opt.iterations:
                with timer.saving():
                    torch.save({'model': model.capture(),'iteration': iteration,'sampler': sampler.capture(),
                        'app':app.state_dict(),'app_optimizer':app.optimizer.state_dict(),
                        'random':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
                        'cuda':torch.cuda.get_rng_state_all()},str(Path(scene.model_path)/f'chkpnt{iteration}.pth'))
            if args.plane_evidence and (iteration == opt.iterations or (args.plane_export_interval>0 and iteration%args.plane_export_interval == 0)):
                from pipeline.planes.export_evidence import export_dynamic_evidence
                export_dynamic_evidence(model,cameras,scene.instance_ids,pipe,background,args,
                    str(Path(scene.model_path)/'dynamic_plane_prior'/f'iteration_{iteration}'))
        del packages,render_models,compact_models,loss
    timer.finish(opt.iterations-first)
    return model,scene


def main():
    parser,model,opt,pipe = make_parser()
    args = parser.parse_args()
    training(model.extract(args),opt.extract(args),pipe.extract(args),args)


if __name__ == '__main__':
    main()
