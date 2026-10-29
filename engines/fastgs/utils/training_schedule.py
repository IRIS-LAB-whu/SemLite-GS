def should_reset_opacity(
    iteration: int,
    opacity_reset_interval: int,
    densify_until_iter: int,
    white_background: bool,
    densify_from_iter: int,
) -> bool:

    reset_cutoff = min(int(densify_until_iter), 15_000)
    periodic_reset = (
        int(iteration) % int(opacity_reset_interval) == 0
        and int(iteration) < reset_cutoff
    )
    white_background_reset = bool(white_background) and int(iteration) == int(
        densify_from_iter
    )
    return periodic_reset or white_background_reset
