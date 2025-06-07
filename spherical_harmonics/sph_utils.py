import torch


def azimuth_elevation_to_cartesian(azimuth, elevation):
    """
    Convert azimuth and elevation angles to cartesian coordinates on the unit sphere.

    Args:
        azimuth (torch.Tensor): Azimuth angles in radians. (batch_size) or (batch_size, n_angles).
        elevation (torch.Tensor): Elevation angles in radians. (batch_size) or (batch_size, n_angles).

    Returns:
        torch.Tensor: Cartesian coordinates. (batch_size, 3) or (batch_size, n_angles, 3).
    """
    assert isinstance(azimuth, torch.Tensor), "Azimuth must be a torch.Tensor"
    assert isinstance(elevation, torch.Tensor), "Elevation must be a torch.Tensor"
    assert (
        azimuth.shape == elevation.shape
    ), "Azimuth and elevation must have the same shape"

    x = torch.cos(azimuth) * torch.cos(elevation)
    y = torch.sin(azimuth) * torch.cos(elevation)
    z = torch.sin(elevation)
    return torch.stack(
        (x, y, z), dim=-1
    )  # (batch_size, 3) or (batch_size, n_angles, 3)
