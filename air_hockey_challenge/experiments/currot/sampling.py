import torch


def to_spherical_coordinates(cartesian_coordinates: torch.Tensor):
    assert len(cartesian_coordinates.shape) == 2
    n, dim = cartesian_coordinates.shape

    polar_coordinates = torch.empty((n, dim - 1), dtype=cartesian_coordinates.dtype,
                                    device=cartesian_coordinates.device)
    for i in range(dim - 2):
        polar_coordinates[:, i] = torch.arccos(
            cartesian_coordinates[:, i] / torch.norm(cartesian_coordinates[:, i:], dim=-1))
    polar_coordinates[:, dim - 2] = torch.arctan2(cartesian_coordinates[:, -1], cartesian_coordinates[:, -2])

    return polar_coordinates


def to_cartesian_coordinates(spherical_coordinates: torch.Tensor):
    assert len(spherical_coordinates.shape) == 2
    n, dim = spherical_coordinates.shape
    # Cartesian coordinates in n-d are represented by n-1 polar coordinates (we assume that these are unit vectors)
    dim = dim + 1

    cum_sins = torch.cumprod(torch.sin(spherical_coordinates), dim=-1)
    coss = torch.cos(spherical_coordinates)
    cartesian_coordinates = torch.empty((n, dim), dtype=spherical_coordinates.dtype,
                                        device=spherical_coordinates.device)
    cartesian_coordinates[:, 0] = coss[:, 0]
    for i in range(1, dim - 1):
        cartesian_coordinates[:, i] = cum_sins[:, i - 1] * coss[:, i]
    cartesian_coordinates[:, -1] = cum_sins[:, -1]

    return cartesian_coordinates


class SphereSampler:
    """
    """

    def __init__(self, dim: int, device="cpu") -> None:
        self.dim = dim
        self.device = device

    def sample_directions(self, n: int):
        directions = torch.randn((n, self.dim), device=self.device)
        directions /= torch.norm(directions, dim=-1, keepdim=True)

        return directions


class MahalanobisSphereSampler(SphereSampler):

    def __init__(self, dim: int, device="cpu", precision_matrix: torch.Tensor = None) -> None:
        super().__init__(dim, device)
        if precision_matrix is None:
            precision_matrix = torch.eye(dim, device=device)
        self.whiten_transform = torch.linalg.cholesky(precision_matrix.double()).T
        self.color_transform = torch.linalg.solve(precision_matrix.double(), self.whiten_transform.T).float()
        self.whiten_transform = self.whiten_transform.float()

    def sample_directions(self, n: int):
        return torch.einsum("ij,...j->...i", self.color_transform, super().sample_directions(n))


class SphericalCapSampler:
    """
    A class that generaTes samples from the surface of a sphere that intersects with a cone defined by an angle alpha
    with the "up-axis" (i.e. the last axis). Taken from: http://www.vldb.org/pvldb/vol12/p237-asudeh.pdf

    I adapted the sampler to be based on the x-axis, since the most commonly discussed coordinate transforms between
    cartesian and spherical put the x-axis as the one from which the cosine is determined, such as wikipedia:
    https://en.wikipedia.org/wiki/N-sphere#Spherical_coordinates
    """

    def __init__(self, dim: int, rad_angle: float, cfd_disc: int = 100, device="cpu", dtype=torch.float32) -> None:
        # We pre-compute the CDF of sampling from a slice of the sphere (which corresponds to a specific angle
        # made with the z-axis (which is the reference axis for computing the angle)
        self.angles = torch.linspace(0, rad_angle, cfd_disc, device=device)
        log_slice_prob = torch.log(torch.sin(self.angles)) * (dim - 2)
        log_slice_prob[0] = -torch.inf
        log_slice_prob -= torch.logsumexp(log_slice_prob, dim=-1)
        self.cdf = torch.cumsum(torch.exp(log_slice_prob), dim=-1)

        self.dim = dim
        self.device = device
        self.dtype = dtype

        # We create an x_axis vector for the rotation operation (we assume that our cone is centered on the x-axis)
        self.x_axis = torch.zeros(self.dim, device=device)
        self.x_axis[0] = 1.

    def sample_directions(self, n: int):
        # Sample n angles using the CDF
        uniform_samples = torch.rand(n, device=self.device, dtype=self.dtype)
        upper_cdf_idxs = torch.argmax((uniform_samples[:, None] < self.cdf[None, :]).float(), dim=-1)
        lower_cdf_idxs = torch.clamp_min(upper_cdf_idxs - 1, 0)

        lower_angles = self.angles[lower_cdf_idxs]
        upper_angles = self.angles[upper_cdf_idxs]
        angles = (upper_angles - lower_angles) * torch.rand(n, device=self.device, dtype=self.dtype) + lower_angles

        # We next sample a uniform vector on the d-1-dimensional hypersphere
        if self.dim == 2:
            polar_coordinates = torch.zeros((n, 0), device=self.device, dtype=self.dtype)
            # If we are in 2D we need to reflect the angle randomly, otherwise we cover only half of the cone
            angles = torch.where(torch.rand(n, dtype=self.dtype, device=self.device) >= 0.5, angles, -angles)
        else:
            sub_vectors = torch.randn((n, self.dim - 1), dtype=self.dtype, device=self.device)
            sub_vectors /= torch.norm(sub_vectors, dim=-1, keepdim=True)

            # We next transform the sub-vector to polar coordinates ...
            polar_coordinates = to_spherical_coordinates(sub_vectors)

        # ... and finally add the sampled final angle
        polar_coordinates = torch.cat((angles[:, None], polar_coordinates), dim=-1)

        # We then transform the spherical coordinates back
        return to_cartesian_coordinates(polar_coordinates)

    def rotate_directions(self, reference_directions: torch.Tensor, search_directions: torch.Tensor):
        # We assume that both tensors how two dimensions where the leading one is the batch dimension
        # We first normalize the directions
        normed_reference_directions = reference_directions / torch.norm(reference_directions, dim=-1, keepdim=True)

        # Next we compute the differences
        v = normed_reference_directions - self.x_axis[None, :]
        norm_v = torch.norm(v, dim=-1, keepdim=True)
        v /= norm_v

        # We will need this mask at the end to filter out nans (which can occur if v is basically the x-axis, i.e. the on which we want to align to)
        # Is of shape (N,)
        mask = torch.where(norm_v[..., 0] < 1e-10)

        # This is an efficient version of the "quick and dirty" way described here
        # https://math.stackexchange.com/questions/525276/rotation-matrix-in-arbitrary-dimension-to-align-vector.
        # The approach is based on computing the householder reflection M = I - 2 * (vv^T) / ||v||^2, then
        # multiplying the last row by -1 and finally using the transpose of the resulting matrix as the required
        # rotation. The following steps are simply an optimized implementation which exploits the structure of M to yield
        # an O(n) algorithm
        flipped_search_directions = search_directions.clone()
        flipped_search_directions[:, -1] *= -1

        rotated_search_directions = flipped_search_directions[None, ...] - 2 * v[:, None, :] * \
                                    torch.einsum("ni,mi->nm", v, flipped_search_directions)[..., None]
        rotated_search_directions[mask] = search_directions[None, ...]
        return rotated_search_directions

    def sample_rotated_directions(self, n: int, reference_directions: torch.Tensor):
        directions = self.sample_directions(n)
        return self.rotate_directions(reference_directions, directions)


class MahalanobisSphericalCapSampler(SphericalCapSampler):

    def __init__(self, dim: int, rad_angle: float, cfd_disc: int = 100, device="cpu", dtype=torch.float32,
                 precision_matrix: torch.Tensor = None) -> None:
        super().__init__(dim, rad_angle, cfd_disc, device, dtype)
        if precision_matrix is None:
            precision_matrix = torch.eye(dim, device=device)
        self.whiten_transform = torch.linalg.cholesky(precision_matrix.double()).T
        self.color_transform = torch.linalg.solve(precision_matrix.double(), self.whiten_transform.T).float()
        self.whiten_transform = self.whiten_transform.float()

    def sample_directions(self, n: int):
        return torch.einsum("ij,...j->...i", self.color_transform, super().sample_directions(n))

    def rotate_directions(self, reference_directions: torch.Tensor, search_directions: torch.Tensor):
        reference_directions = torch.einsum("ij,...j->...i", self.whiten_transform, reference_directions)
        search_directions = torch.einsum("ij,...j->...i", self.whiten_transform, search_directions)
        return torch.einsum("ij,...j->...i", self.color_transform,
                            super().rotate_directions(reference_directions, search_directions))

    def sample_rotated_directions(self, n: int, reference_directions: torch.Tensor):
        directions = super().sample_directions(n)
        reference_directions = torch.einsum("ij,...j->...i", self.whiten_transform, reference_directions)
        return torch.einsum("ij,...j->...i", self.color_transform,
                            super().rotate_directions(reference_directions, directions))


def main1():
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.spatial.transform import Rotation

    # Check the polar coordinate transform
    for dim in range(2, 100):
        cartesian_coords = torch.randn((500, dim)).double()
        cartesian_coords /= torch.norm(cartesian_coords, dim=-1, keepdim=True)

        polar_coords = to_spherical_coordinates(cartesian_coords)
        rec_cartesian_coords = to_cartesian_coordinates(polar_coords)

        assert torch.allclose(cartesian_coords, rec_cartesian_coords)

    # Create a sampler
    angle = 0.1 * torch.pi
    twod_sampler = SphericalCapSampler(2, angle)
    threed_sampler = SphericalCapSampler(3, angle)
    threed_sphere_sampler = SphereSampler(3)

    directions_2d = twod_sampler.sample_directions(200)
    directions_3d = threed_sampler.sample_directions(200)

    plt.plot(np.sin(np.linspace(0, 2 * np.pi, 500)), np.cos(np.linspace(0, 2 * np.pi, 500)), linewidth=2)
    plt.scatter(directions_2d[:, 0], directions_2d[:, 1])
    plt.scatter(np.cos(angle), np.sin(angle), color="C1")
    plt.scatter(np.cos(-angle), np.sin(-angle), color="C1")

    ref_dir = torch.randn(size=(2,))
    ref_dir /= torch.norm(ref_dir)

    rotated_directions_2d = twod_sampler.rotate_directions(ref_dir[None], directions_2d)[0]
    plt.plot([0, ref_dir[0]], [0, ref_dir[1]], color="C4", linewidth=2)
    plt.scatter(rotated_directions_2d[:, 0], rotated_directions_2d[:, 1], color="C4")

    plt.gca().set_aspect('equal')
    plt.tight_layout()
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    sphere_samples = threed_sphere_sampler.sample_directions(1000)

    # Plot the samples from the sphere and the cone
    ax.scatter(sphere_samples[:, 0], sphere_samples[:, 1], sphere_samples[:, 2], s=10)
    ax.scatter(directions_3d[:, 0], directions_3d[:, 1], directions_3d[:, 2], color="C1", s=30)

    # Plot some of the rays that intersect with the cone
    cone_vecs = np.squeeze(np.stack([Rotation.from_euler("y", [angle], degrees=False).apply(np.array([1., 0., 0.])),
                                     Rotation.from_euler("y", [-angle], degrees=False).apply(np.array([1., 0., 0.])),
                                     Rotation.from_euler("z", [angle], degrees=False).apply(np.array([1., 0., 0.])),
                                     Rotation.from_euler("z", [-angle], degrees=False).apply(np.array([1., 0., 0.]))]))
    cone_vecs *= 1.25

    ref_dir = torch.randn(size=(3,))
    ref_dir /= torch.norm(ref_dir)

    rotated_directions_3d = threed_sampler.rotate_directions(ref_dir[None], directions_3d)[0]

    for cv in cone_vecs:
        ax.plot([0, cv[0]], [0, cv[1]], zs=[0, cv[2]], color="C1", linewidth=3)

    ax.plot([0, ref_dir[0]], [0, ref_dir[1]], zs=[0, ref_dir[2]], color="C4", linewidth=3)
    ax.scatter(rotated_directions_3d[:, 0], rotated_directions_3d[:, 1], rotated_directions_3d[:, 2], color="C4", s=30)

    ax.set_box_aspect((2, 2, 2))
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    plt.show()

    # Quick check out of curiosity - is a 45 degree angle always guaranteeing a descent direction
    for dim in range(4, 200):
        source_points = 2 * torch.rand((1280, dim)) - 1
        target_points = 2 * torch.rand((1280, dim)) - 1

        sampler = SphericalCapSampler(dim, 0.25 * torch.pi)
        search_dirs = sampler.sample_directions(50)

        ref_dirs = target_points - source_points
        init_dist = torch.norm(ref_dirs, dim=-1)
        steps = sampler.rotate_directions(ref_dirs, search_dirs)

        step_sizes = torch.minimum(torch.ones(1), init_dist)
        steps *= step_sizes[:, None, None]

        new_dist = torch.norm(target_points[:, None, :] - (source_points[:, None, :] + steps), dim=-1)
        assert torch.all(new_dist <= init_dist[:, None])

    print("45 degree is a descent direction across dimensions!")

    # Finally, we verify the sampler for high-d spaces by checking that the sampled points fulfill the angle requirement with the x-axis
    for dim in range(4, 200):
        nd_sampler = SphericalCapSampler(dim, rad_angle=angle, dtype=torch.float)
        dirs = nd_sampler.sample_directions(100)
        dist_mat = torch.norm(dirs[:, None] - dirs[None, :], dim=-1)

        assert torch.allclose(torch.norm(dirs, dim=-1), torch.ones(1))
        assert torch.all(torch.abs(torch.arccos(torch.einsum("ni,i->n", dirs, nd_sampler.x_axis))) < angle)

        # Try a transform to another random direction
        ref_dirs = SphereSampler(dim).sample_directions(20)
        rotated_dirs = nd_sampler.rotate_directions(ref_dirs, dirs)
        rotated_dist_mat = torch.norm(rotated_dirs[:, :, None, :] - rotated_dirs[:, None, :, :], dim=-1)

        assert torch.allclose(rotated_dist_mat, dist_mat[None, ...])
        assert torch.allclose(torch.norm(rotated_dirs, dim=-1), torch.ones(1))
        assert torch.all(torch.abs(torch.arccos(torch.einsum("nmi,ni->nm", rotated_dirs, ref_dirs))) < angle + 1e-4)


def main2():
    import matplotlib.pyplot as plt

    precision_matrix = torch.linalg.inv(torch.tensor([[0.5, -0.5],
                                                      [-0.5, 2.]]))
    twod_sampler = MahalanobisSphericalCapSampler(2, rad_angle=0.1 * torch.pi,
                                                  precision_matrix=precision_matrix)

    # Plot the matrix
    whitened_circle = torch.stack((torch.sin(torch.linspace(0, 2 * torch.pi, 500)),
                                   torch.cos(torch.linspace(0, 2 * torch.pi, 500))), dim=-1)
    colored_circle = torch.einsum("ij,...j->...i", twod_sampler.color_transform, whitened_circle)

    colored_directions = twod_sampler.sample_directions(100)
    whitened_directions = torch.einsum('ij,...j->...i', twod_sampler.whiten_transform, colored_directions)

    idx = torch.randint(colored_circle.shape[0], (1,))
    colored_ref_dir = colored_circle[idx][0]
    whitened_ref_dir = whitened_circle[idx][0]
    # colored_search_directions = twod_sampler.rotate_directions(colored_ref_dir[None, :], colored_directions)[0]
    colored_search_directions = twod_sampler.sample_rotated_directions(100, colored_ref_dir[None, :])[0]
    whitened_search_directions = torch.einsum("ij,...j->...i", twod_sampler.whiten_transform, colored_search_directions)

    plt.plot(whitened_circle[:, 0], whitened_circle[:, 1])
    plt.scatter(whitened_directions[:, 0], whitened_directions[:, 1])
    plt.scatter(whitened_search_directions[:, 0], whitened_search_directions[:, 1], color="C4")
    plt.plot([0, whitened_ref_dir[0]], [0, whitened_ref_dir[1]], color="C4", linewidth=2)
    plt.gca().set_aspect('equal')
    plt.show()

    plt.plot(colored_circle[:, 0], colored_circle[:, 1])
    plt.scatter(colored_directions[:, 0], colored_directions[:, 1])
    plt.scatter(colored_search_directions[:, 0], colored_search_directions[:, 1], color="C4")
    plt.plot([0, colored_ref_dir[0]], [0, colored_ref_dir[1]], color="C4", linewidth=2)
    plt.gca().set_aspect('equal')
    plt.show()


if __name__ == "__main__":
    main1()
    # main2()
