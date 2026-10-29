from pathlib import Path
import torch
from scene import Scene as NativeScene
from pipeline.masks import load_instance_mask, infer_instance_ids


class Scene(NativeScene):
    def __init__(self, args, gaussians, pipeline_args, **kwargs):
        Path(args.model_path).mkdir(parents=True, exist_ok=True)
        from copy import copy
        native_args = copy(args)
        native_args.preload_img = False
        sparse = Path(args.source_path)/'sparse'
        if (sparse/'0').is_dir() and not any((sparse/name).is_file() for name in ('cameras.bin','cameras.txt')):
            from copy import copy
            native_args = copy(args)
            native_args.preload_img = False
            staging = Path(args.model_path).resolve()/'native_input'
            staging.mkdir(exist_ok=True)
            links = {'sparse': sparse.resolve()/'0', args.images: Path(args.source_path).resolve()/args.images}
            for name,target in links.items():
                link = staging/name
                if not link.exists():
                    link.symlink_to(target,target_is_directory=True)
            split = Path(args.source_path)/'split.json'
            if split.exists() and not (staging/'split.json').exists():
                (staging/'split.json').symlink_to(split.resolve())
            native_args.source_path = str(staging)
        if min(getattr(pipeline_args, 'camera_io_workers', 0),
               getattr(pipeline_args, 'guard_io_workers', 0)) < 0:
            raise ValueError('IO workers must be nonnegative')
        native_args._pipeline_skip_pcd_init = bool(
            getattr(pipeline_args, 'fast_init_ply', False) and
            getattr(pipeline_args, 'init_ply', '') and not kwargs.get('load_iteration'))
        super().__init__(native_args, gaussians, **kwargs)
        gaussians.fast_point_budget = getattr(pipeline_args, 'fast_point_budget', False)
        self.source_path = args.source_path
        if getattr(pipeline_args, 'init_ply', '') and not kwargs.get('load_iteration'):
            gaussians.load_ply(pipeline_args.init_ply)
            shift = getattr(pipeline_args, 'init_ply_translation', None)
            if shift is not None:
                if len(shift) != 3 or not torch.isfinite(torch.tensor(shift)).all():
                    raise ValueError('init_ply_translation requires three finite coordinates')
                with torch.no_grad():
                    gaussians._xyz.add_(torch.as_tensor(shift, device=gaussians._xyz.device,
                                                        dtype=gaussians._xyz.dtype))
        mask_root = Path(pipeline_args.instance_masks) if pipeline_args.instance_masks else None
        if mask_root is not None and not mask_root.is_absolute():
            mask_root = Path(args.source_path) / mask_root
        cameras = self.getTrainCameras() + self.getTestCameras()
        from pipeline.cameras import restore_colmap_principal_points
        restore_colmap_principal_points(cameras,args.source_path)
        from pipeline.cameras import configure_lazy_cameras
        configure_lazy_cameras(cameras,mask_root,args.model_path,pipeline_args.image_cache_size,
                               min_mask_area=getattr(pipeline_args,'min_instance_mask_area',200),
                               io_workers=getattr(pipeline_args,'camera_io_workers',0))
        raw_ids = pipeline_args.instance_ids
        self.instance_ids = [int(v) for v in raw_ids.split(',') if v] if raw_ids else infer_instance_ids(cameras,pipeline_args.mask_background_id)
        if not self.instance_ids:
            if mask_root:
                raise ValueError('No foreground instance labels found')
            self.instance_ids = [1]


        if kwargs.get('load_iteration'):
            self.prism_models = {}
            self.final_prism_models = {}
            return
        owners = torch.full((len(gaussians.get_xyz),), -1, device=gaussians.get_xyz.device, dtype=torch.long)
        self.prism_models = {}
        self.final_prism_models = {}
        if pipeline_args.building_ply:
            from pipeline.planes.building_prisms import build_building_prism_model, points_in_prisms_numpy
            if not pipeline_args.building_transform:
                raise ValueError('--building_ply requires --building_transform')
            for instance in self.instance_ids:
                prism = build_building_prism_model([instance], pipeline_args.building_ply,
                    pipeline_args.building_transform, value_mapping_path=pipeline_args.building_values,
                    color_mapping_path=pipeline_args.building_colors,
                    xy_padding=getattr(pipeline_args,'global_freeze_xy_padding',0.),
                    z_padding=getattr(pipeline_args,'global_freeze_z_padding',0.),
                    trim_quantile=getattr(pipeline_args,'global_freeze_trim_quantile',.02))
                self.prism_models[instance] = prism
                if getattr(pipeline_args,'prune_final_to_building_prisms',False):
                    self.final_prism_models[instance] = build_building_prism_model([instance],pipeline_args.building_ply,
                        pipeline_args.building_transform,value_mapping_path=pipeline_args.building_values,
                        color_mapping_path=pipeline_args.building_colors,
                        xy_padding=getattr(pipeline_args,'final_prune_xy_padding',.5),
                        z_padding=getattr(pipeline_args,'final_prune_z_padding',.5),
                        trim_quantile=getattr(pipeline_args,'final_prune_trim_quantile',.005))
                inside = torch.as_tensor(points_in_prisms_numpy(gaussians.get_xyz.detach().cpu().numpy(), prism), device=owners.device)
                owners[(owners < 0) & inside] = instance
        elif mask_root:

            votes = torch.zeros((len(self.instance_ids), len(owners)), dtype=torch.int32, device=owners.device)
            for camera in self.getTrainCameras():
                xyz = gaussians.get_xyz.detach() @ camera.world_view_transform[:3,:3] + camera.world_view_transform[3,:3]
                u = (xyz[:,0]*camera.Fx/xyz[:,2]+camera.Cx).round().long()
                v = (xyz[:,1]*camera.Fy/xyz[:,2]+camera.Cy).round().long()
                valid = (xyz[:,2]>0) & (u>=0) & (u<camera.image_width) & (v>=0) & (v<camera.image_height)
                labels = camera.instance_mask.to(owners.device)[v[valid],u[valid]]
                for j,instance in enumerate(self.instance_ids):
                    votes[j,valid] += (labels == instance).int()
            best, index = votes.max(dim=0)
            owners[best>0] = torch.tensor(self.instance_ids, device=owners.device)[index[best>0]]
        else:
            owners.fill_(1)
        if not kwargs.get('load_iteration'):
            gaussians.initialize_metadata(owners, pipeline_args.freeze_global_outside_instances,
                                         pipeline_args.max_points_per_instance)
        if pipeline_args.freeze_global_outside_instances and not gaussians.trainable_mask.any():
            raise ValueError('No trainable Gaussian lies inside selected original instances/prisms')


    def save(self, iteration, mask=None):

        import json
        super().save(iteration,mask)
        root=Path(self.model_path)/'point_cloud'/f'iteration_{iteration}'
        m=self.gaussians
        included=torch.ones(len(m.get_xyz),dtype=torch.bool,device=m.get_xyz.device) if mask is None else mask.bool().to(m.get_xyz.device)
        summary={'context_mode': m.context_mode, 'total_points':int(included.sum()),'trainable_points':int((included & m.trainable_mask).sum()),
                 'frozen_points':int((included & ~m.trainable_mask).sum()),
                 'context_cloud':'point_cloud.ply','instances':{}}
        for instance in self.instance_ids:
            selected=included & (m.instance_ids==instance)
            filename=f'point_cloud_instance_{instance}.ply'
            m.save_ply(str(root/filename),selected)
            summary['instances'][str(instance)]={'points':int(selected.sum()),'file':filename}
        (root/'point_cloud_summary.json').write_text(json.dumps(summary,indent=2))
