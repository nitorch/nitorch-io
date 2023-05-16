"""Various functions that relate to affine spaces.

NiTorch encodes affine matrices using their Lie algebra representation.
This
"""

# TODO:
#   * Batch ``volume_axis`` and ``volume_layout``
#     -> need to find a way to differentiate "index" from "axis" inputs.

import torch
from . import py, linalg, itertools, torchutils as utils
import math


def volume_axis(*args, **kwargs):
    """Describe an axis of a volume of voxels.

    Signature
    ---------
    * ``def volume_axis(index, flipped=False, device=None)``
    * ``def volume_axis(name, device=None)``
    * ``def volume_axis(axis, device=None)``

    Parameters
    ----------
    index : () tensor_like[int]
        Index of the axis in 'direct' space (RAS)

    flipped : () tensor_like[bool], default=False
        Whether the axis is flipped or not.

    name : {'R' or 'L', 'A' or 'P', 'S' or 'I'}
        Name of the axis, according to the neuroimaging convention:
        * 'R' for *left to Right* (index=0, flipped=False) or
          'L' for *right to Left* (index=0, flipped=True)
        * 'A' for *posterior to Anterior* (index=1, flipped=False) or
          'P' for *anterior to Posterior* (index=1, flipped=True)
        * 'S' for *inferior to Superior* (index=2, flipped=False) or
          'I' for *superior to Inferior* (index=2, flipped=True)

    axis : (2,) tensor_like[long]
        ax[0] = index
        ax[1] = flipped
        Description of the axis.

    device : str or torch.device
        Device.

    Returns
    -------
    axis : (2,) tensor[long]
        Description of the axis.
        * `ax[0] = index`
        * `ax[1] = flipped`

    """
    def axis_from_name(name, device=None):
        name = name.upper()
        if name == 'R':
            return torch.as_tensor([0, 0], dtype=torch.long, device=device)
        elif name == 'L':
            return torch.as_tensor([0, 1], dtype=torch.long, device=device)
        elif name == 'A':
            return torch.as_tensor([1, 0], dtype=torch.long, device=device)
        elif name == 'P':
            return torch.as_tensor([1, 1], dtype=torch.long, device=device)
        elif name == 'S':
            return torch.as_tensor([2, 0], dtype=torch.long, device=device)
        elif name == 'I':
            return torch.as_tensor([2, 1], dtype=torch.long, device=device)

    def axis_from_index(index, flipped=False, device=None):
        index = utils.as_tensor(index).reshape(())
        flipped = utils.as_tensor(flipped).reshape(())
        return utils.as_tensor([index, flipped], dtype=torch.long, device=device)

    def axis_from_axis(ax, device=None):
        ax = utils.as_tensor(ax, dtype=torch.long, device=device).flatten()
        if ax.numel() != 2:
            raise ValueError('An axis should have two elements. Got {}.'
                             .format(ax.numel()))
        return ax

    # Dispatch based on input types or keyword arguments
    args = list(args)
    if len(args) > 0:
        if isinstance(args[0], str):
            return axis_from_name(*args, **kwargs)
        else:
            args[0] = utils.as_tensor(args[0])
            if args[0].numel() == 1:
                return axis_from_index(*args, **kwargs)
            else:
                return axis_from_axis(*args, **kwargs)
    else:
        if 'name' in kwargs.keys():
            return axis_from_name(*args, **kwargs)
        elif 'index' in kwargs.keys():
            return axis_from_index(*args, **kwargs)
        else:
            return axis_from_axis(*args, **kwargs)


# Mapping from (index, flipped) to axis name
_axis_names = [['R', 'L'], ['A', 'P'], ['S', 'I']]


def volume_axis_to_name(axis):
    """Return the (neuroimaging) name of an axis. Its index must be < 3.

    Parameters
    ----------
    axis : (2,) tensor_like

    Returns
    -------
    name : str

    """
    index, flipped = axis
    if index >= 3:
        raise ValueError('Index names are only defined up to dimension 3. '
                         'Got {}.'.format(index))
    return _axis_names[index][flipped]


def volume_layout(*args, **kwargs):
    """Describe the layout of a volume of voxels.

    A layout is characterized by a list of axes. See `volume_axis`.

    Signature
    ---------
    volume_layout(dim=3, device=None)
    volume_layout(name, device=None)
    volume_layout(axes, device=None)
    volume_layout(index, flipped=False, device=None)

    Parameters
    ----------
    dim : int, default=3
        Dimension of the space.
        This version of the function always returns a directed layout
        (identity permutation, no flips), which is equivalent to 'RAS'
        but in any dimension.

    name : str
        Permutation of axis names, according to the neuroimaging convention:
        * 'R' for *left to Right* (index=0, flipped=False) or
          'L' for *right to Left* (index=0, flipped=True)
        * 'A' for *posterior to Anterior* (index=1, flipped=False) or
          'P' for *anterior to Posterior* (index=1, flipped=True)
        * 'S' for *inferior to Superior* (index=2, flipped=False) or
          'I' for *superior to Inferior* (index=2, flipped=True)
        The number of letters defines the dimension of the matrix
        (`ndim = len(name)`).

    axes : (ndim, 2) tensor_like[long]
        List of objects returned by `axis`.

    index : (ndim, ) tensor_like[long]
        Index of the axes in 'direct' space (RAS)

    flipped : (ndim, ) tensor_like[bool], default=False
        Whether each axis is flipped or not.

    Returns
    -------
    layout : (ndim, 2) tensor[long]
        Description of the layout.

    """
    def layout_from_dim(dim, device=None):
        return volume_layout(list(range(dim)), flipped=False, device=device)

    def layout_from_name(name, device=None):
        return volume_layout([volume_axis(a, device) for a in name])

    def layout_from_index(index, flipped=False, device=None):
        index = utils.as_tensor(index, torch.long, device).flatten()
        ndim = index.shape[0]
        flipped = utils.as_tensor(flipped, torch.long, device).flatten()
        if flipped.shape[0] == 1:
            flipped = torch.repeat_interleave(flipped, ndim, dim=0)
        return torch.stack((index, flipped), dim=-1)

    def layout_from_axes(axes, device=None):
        axes = utils.as_tensor(axes, torch.long, device)
        if axes.dim() != 2 or axes.shape[1] != 2:
            raise ValueError('A layout should have shape (ndim, 2). Got {}.'
                             .format(axes.shape))
        return axes

    # Dispatch based on input types or keyword arguments
    args = list(args)
    if len(args) > 0:
        if isinstance(args[0], str):
            layout = layout_from_name(*args, **kwargs)
        else:
            args[0] = utils.as_tensor(args[0])
            if args[0].dim() == 0:
                layout = layout_from_dim(*args, **kwargs)
            elif args[0].dim() == 2:
                layout = layout_from_axes(*args, **kwargs)
            else:
                layout = layout_from_index(*args, **kwargs)
    else:
        if 'dim' in kwargs.keys():
            layout = layout_from_dim(*args, **kwargs)
        elif 'name' in kwargs.keys():
            layout = layout_from_name(*args, **kwargs)
        elif 'index' in kwargs.keys():
            layout = layout_from_index(*args, **kwargs)
        else:
            layout = layout_from_axes(*args, **kwargs)

    # Remap axes indices if not contiguous
    backend = dict(dtype=layout.dtype, device=layout.device)
    axes = layout[:, 0]
    remap = torch.argsort(axes)
    axes[remap] = torch.arange(len(axes), **backend)
    layout = torch.stack((axes, layout[:, 1]), dim=-1)
    return layout


def volume_layout_to_name(layout):
    """Return the (neuroimaging) name of a layout.

    Its length must be <= 3 (else, we just return the permutation and
    flips), e.g. '[2, 3, -1, 4]'

    Parameters
    ----------
    layout : (dim, 2) tensor_like

    Returns
    -------
    name : str

    """
    layout = volume_layout(layout)
    if len(layout) > 3:
        layout = [('-' if bool(f) else '') + str(int(p)) for p, f in layout]
        return '[' + ', '.join(layout) + ']'
    names = [volume_axis_to_name(axis) for axis in layout]
    return ''.join(names)


def iter_layouts(ndim, device=None):
    """Compute all possible layouts for a given dimensionality.

    Parameters
    ----------
    ndim : () tensor_like
        Dimensionality (rank) of the space.

    Returns
    -------
    layouts : (nflip*nperm, ndim, 2) tensor[long]
        All possible layouts.
        * nflip = 2 ** ndim     -> number of flips
        * nperm = ndim!         -> number of permutations

    """
    if device is None:
        if torch.is_tensor(ndim):
            device = ndim.device
    backend = dict(dtype=torch.long, device=device)

    # First, compute all possible directed layouts on one hand,
    # and all possible flips on the other hand.
    axes = torch.arange(ndim, **backend)
    layouts = itertools.permutations(axes)                      # [P, D]
    flips = itertools.product([0, 1], r=ndim, **backend)        # [F, D]

    # Now, compute combination (= cartesian product) of both
    # We replicate each tensor so that shapes match and stack them.
    nb_layouts = len(layouts)
    nb_flips = len(flips)
    layouts = layouts[None, ...]
    layouts = torch.repeat_interleave(layouts, nb_flips, dim=0)  # [F, P, D]
    flips = flips[:, None, :]
    flips = torch.repeat_interleave(flips, nb_layouts, dim=1)    # [F, P, D]
    layouts = torch.stack([layouts, flips], dim=-1)

    # Finally, flatten across repeats
    layouts = layouts.reshape([-1, ndim, 2])    # [F*P, D, 2]

    return layouts


def layout_matrix(layout, voxel_size=1., shape=0., dtype=None, device=None):
    """Compute the origin affine matrix for different voxel layouts.

    Resources
    ---------
    .. https://nipy.org/nibabel/image_orientation.html
    .. https://nipy.org/nibabel/neuro_radio_conventions.html

    Parameters
    ----------
    layout : str or (ndim, 2) tensor_like[long]
        See `affine.layout`

    voxel_size : (ndim,) tensor_like, default=1
        Voxel size of the lattice

    shape : (ndim,) tensor_like, default=0
        Shape of the lattice

    dtype : torch.dtype, optional
        Data type of the matrix

    device : torch.device, optional
        Output device.

    Returns
    -------
    mat : (ndim+1, ndim+1) tensor[dtype]
        Corresponding affine matrix.

    """

    # Author
    # ------
    # .. Yael Balbastre <yael.balbastre@gmail.com>

    # Extract info from layout
    layout = volume_layout(layout, device=device)
    device = layout.device
    dim = len(layout)
    perm = utils.invert_permutation(layout[:, 0])
    flip = layout[:, 1].bool()

    # ensure tensor
    voxel_size = torch.as_tensor(voxel_size, dtype=dtype, device=device)
    dtype = voxel_size.dtype
    shape = torch.as_tensor(shape, dtype=dtype, device=device)

    # ensure dim
    shape = utils.ensure_shape(shape, [dim], mode='replicate')
    voxel_size = utils.ensure_shape(voxel_size, [dim], mode='replicate')
    zero = torch.zeros(dim, dtype=dtype, device=device)

    # Create matrix
    mat = torch.diag(voxel_size)
    mat = mat[perm, :]
    mat = torch.cat((mat, zero[:, None]), dim=1)
    mflip = torch.ones(dim, dtype=dtype, device=device)
    mflip = torch.where(flip, -mflip, mflip)
    mflip = torch.diag(mflip)
    shift = torch.where(flip, shape[perm], zero)
    mflip = torch.cat((mflip, shift[:, None]), dim=1)
    mat = affine_matmul(as_euclidean(mat), as_euclidean(mflip))
    mat = affine_make_homogeneous(mat)

    return mat


def affine_to_layout(mat):
    """Find the volume layout associated with an affine matrix.

    Parameters
    ----------
    mat : (..., dim, dim+1) or (..., dim+1, dim+1) tensor_like
        Affine matrix (or matrices).

    Returns
    -------
    layout : (..., dim, 2) tensor[long]
        Volume layout(s) (see `volume_layout`).

    """

    # Authors
    # -------
    # .. John Ashburner <j.ashburner@ucl.ac.uk> : original idea
    # .. Yael Balbastre <yael.balbastre@gmail.com> : Python code

    # Convert input
    mat = utils.as_tensor(mat)
    dtype = mat.dtype
    device = mat.device

    # Extract linear component + remove voxel scaling
    dim = mat.shape[-1] - 1
    mat = mat[..., :dim, :dim]
    vs = (mat ** 2).sum(dim=-1)
    mat = linalg.rmdiv(mat, torch.diag(vs))
    eye = torch.eye(dim, dtype=dtype, device=device)

    # Compute SOS between a layout matrix and the (stripped) affine matrix
    def check_space(space):
        layout = layout_matrix(space)[:dim, :dim]
        sos = ((linalg.rmdiv(mat, layout) - eye) ** 2).sum()
        return sos

    # Compute SOS between each layout and the (stripped) affine matrix
    all_layouts = iter_layouts(dim)
    all_sos = torch.stack([check_space(layout) for layout in all_layouts])
    argmin_layout = torch.argmin(all_sos, dim=0)
    min_layout = all_layouts[argmin_layout, ...]

    return min_layout


def as_homogeneous(affine):
    """Mark a tensor as having homogeneous coordinates.

    We set the attribute `is_homogeneous` to True.
    Note that this attribute is not persistent and will be lost
    after any operation on the tensor.
    """
    affine.is_homogeneous = True
    return affine


def as_euclidean(affine):
    """Mark a tensor as having non-homogeneous coordinates.

    We set the attribute `is_homogeneous` to True.
    Note that this attribute is not persistent and will be lost
    after any operation on the tensor.
    """
    affine.is_homogeneous = False
    return affine


def affine_is_square(affine):
    """Return True if the matrix is square"""
    affine = torch.as_tensor(affine)
    return affine.shape[-1] == affine.shape[-2]


def affine_is_rect(affine):
    """Return False if the matrix is rectangular"""
    affine = torch.as_tensor(affine)
    return affine.shape[-1] == affine.shape[-2] + 1


def is_homogeneous(affine):
    """Return true is the last row is [*zeros, 1]"""
    return getattr(affine, 'is_homogeneous', True)


def affine_make_square(affine):
    """Transform a rectangular affine into a square affine.

    Parameters
    ----------
    affine : (..., ndim[+1], ndim+1) tensor

    Returns
    -------
    affine : (..., ndim+1, ndim+1) tensor

    """
    affine = torch.as_tensor(affine)
    device = affine.device
    dtype = affine.dtype
    ndims = affine.shape[-1]-1
    if affine.shape[-2] not in (ndims, ndims+1):
        raise ValueError('Input affine matrix should be of shape\n'
                         '(..., ndims+1, ndims+1) or (..., ndims, ndims+1).')
    if affine.shape[-1] != affine.shape[-2]:
        bottom_row = torch.cat((torch.zeros(ndims, device=device, dtype=dtype),
                                torch.ones(1, device=device, dtype=dtype)), dim=0)
        bottom_row = bottom_row.unsqueeze(0)
        bottom_row = bottom_row.expand(affine.shape[:-2] + bottom_row.shape)
        affine = torch.cat((affine, bottom_row), dim=-2)
    return affine


def affine_make_rect(affine):
    """Transform a square affine into a rectangular affine.

    Parameters
    ----------
    affine : (..., ndim[+1], ndim+1) tensor

    Returns
    -------
    affine : (..., ndim, ndim+1) tensor

    """
    affine = torch.as_tensor(affine)
    ndims = affine.shape[-1]-1
    return affine[..., :ndims, :]


def affine_make_homogeneous(affine):
    """Ensure that the last row of the matrix is (*zeros, 1).

    This function is more generic than `make_square` because it
    works with images where the dimension of the output space differs
    from the dimension of the input space.

    Parameters
    ----------
    affine : (..., ndim_out[+1], ndim_in+1) tensor
        Input affine matrix.

        To inform this function that the input matrix is not homogeneous,
        `as_euclidean` should be called on it beforehand, e.g.,
        `affine_make_homogeneous(as_euclidean(mat))`.

    Returns
    -------
    affine : (..., ndim_out+1, ndim_in+1) tensor

    """
    if is_homogeneous(affine):
        return affine
    affine = torch.as_tensor(affine)
    new_shape = list(affine.shape)
    new_shape[-2] += 1
    new_affine = affine.new_zeros(new_shape)
    new_affine[..., :-1, :] = affine
    new_affine[..., -1, -1] = 1
    return as_homogeneous(new_affine)


def affine_del_homogeneous(affine):
    """Ensure that the last row of the matrix is _not_ (*zeros, 1).

    This function is more generic than `make_rect` because it
    works with images where the dimension of the output space differs
    from the dimension of the input space.

    Parameters
    ----------
    affine : (..., ndim_out[+1], ndim_in+1) tensor
        Input affine matrix

        To inform this function that the input matrix is not homogeneous,
        `as_euclidean` should be called on it beforehand, e.g.,
        `affine_make_homogeneous(as_euclidean(mat))`.

    Returns
    -------
    affine : (..., ndim_out, ndim_in+1) tensor

    """
    if not is_homogeneous(affine):
        return affine
    affine = torch.as_tensor(affine)
    affine = affine[..., :-1, :]
    return as_euclidean(affine)


def voxel_size(mat):
    """ Compute voxel sizes from affine matrices.

    Parameters
    ----------
    mat :  (..., ndim_out[+1], ndim_in+1) tensor
        Affine matrix.

        If the matrix is not homogeneous (i.e., the last row [*zeros, 1]
        is omitted), it must have the attribute `is_homogeneous = False`.
        This attribute can be set on the input tensor by calling
        `as_euclidean` on it.

    Returns
    -------
    vx :  (..., ndim_in) tensor
        Voxel size

    """
    mat = torch.as_tensor(mat)
    if is_homogeneous(mat):
        mat = mat[..., :-1, :-1]
    else:
        mat = mat[..., :, :-1]
    return as_euclidean(mat.square().sum(-2).sqrt())


_voxel_size = voxel_size   # useful alias to avoid shadowing


def affine_matvec(affine, vector):
    """Matrix-vector product of an affine and a (possibly homogeneous) vector.

    Parameters
    ----------
    affine : (..., ndim_out[+1], ndim_in+1) tensor
        Affine matrix.
        If the matrix is not homogeneous (i.e., the last row [*zeros, 1]
        is omitted), it must have the attribute `is_homogeneous = False`.
        This attribute can be set on the input tensor by calling
        `as_euclidean` on it.
    vector : (..., ndim_in[+1]) tensor
        The vector can be homogeneous or not (it is detected
        automatically). The output will be of the same type
        (homogeneous or euclidean) as `vector`.

    Returns
    -------
    affine_times_vector : (..., ndim_out[+1]) tensor
        The returned vector is homogeneous iff `vector` is too.

    """
    affine = affine_del_homogeneous(affine)
    vector = torch.as_tensor(vector)
    backend = dict(dtype=vector.dtype, device=vector.device)
    affine = affine.to(**backend)
    ndims_in = affine.shape[-1] - 1
    is_h = vector.shape[-1] == ndims_in + 1
    zoom = affine[..., :, :-1]
    translation = affine[..., :, -1]
    out = linalg.matvec(zoom, vector[..., :ndims_in]) + translation
    if is_h:
        one = torch.ones(out.shape[:-1] + (1,), **backend)
        out = torch.cat((out, one), dim=-1)
        out = as_homogeneous(out)
    else:
        out = as_euclidean(out)
    return out


def affine_matmul(a, b):
    """Matrix-matrix product of affine matrices.

    Parameters
    ----------
    a : (..., ndim_out[+1], ndim_inter+1) tensor
        Affine matrix
    b : (..., ndim_inter[+1], ndim_in+1) tensor
        Affine matrix

    Returns
    -------
    affine_times_matrix : (..., ndim_out[+1], ndim_in+1) tensor
        The returned matrix is homogeneous iff `a` is too.

    """
    is_h = is_homogeneous(a)
    a = affine_del_homogeneous(a)
    b = affine_del_homogeneous(b)
    a, b = utils.to_max_backend(a, b)
    Za = a[..., :, :-1]
    Ta = a[..., :, -1]
    Zb = b[..., :, :-1]
    Tb = b[..., :, -1]
    Z = torch.matmul(Za, Zb)
    T = linalg.matvec(Za, Tb) + Ta
    out = torch.cat((Z, T[..., None]), dim=-1)
    out = as_euclidean(out)
    if is_h:
        out = affine_make_homogeneous(out)
    return out


def affine_inv(affine):
    """Inverse of an affine matrix.

    If the input matrix is not symmetric with respect to its input
    and output spaces, a pseudo-inverse is returned instead.

    Parameters
    ----------
    affine : (..., ndim_out[+1], ndim_in+1) tensor
        Input affine

    Returns
    -------
    inv_affine : (..., ndim_in[+1], ndim_out+1) tensor
        The returned matrix is homogeneous iff `affine` is too.

    """
    is_h = is_homogeneous(affine)
    affine = affine_del_homogeneous(affine)
    ndims_in = affine.shape[-1] - 1
    ndims_out = affine.shape[-2]
    inverse = torch.inverse if ndims_in == ndims_out else torch.pinverse
    zoom = inverse(affine[..., :, :-1])
    translation = -linalg.matvec(zoom, affine[..., :, -1])
    out = torch.cat((zoom, translation[..., None]), dim=-1)
    out = as_euclidean(out)
    if is_h:
        out = affine_make_homogeneous(out)
    return out


def affine_lmdiv(a, b):
    """Left matrix division of affine matrices: inv(a) @ b

    Parameters
    ----------
    a : (..., ndim_inter[+1], ndim_out+1) tensor
        Affine matrix that will be inverted.
        A peudo-inverse is used if `ndim_inter != ndim_out`
    b : (..., ndim_inter[+1], ndim_in+1) tensor
        Affine matrix.

    Returns
    -------
    output_affine : (..., ndim_out[+1], ndim_in+1) tensor
        The returned matrix is homogeneous iff `a` is too.

    """
    return affine_matmul(affine_inv(a), b)


def affine_rmdiv(a, b):
    """Right matrix division of affine matrices: a @ inv(b)

    Parameters
    ----------
    a : (..., ndim_out[+1], ndim_inter+1) tensor
        Affine matrix.
    b : (..., ndim_in[+1], ndim_inter+1) tensor
        Affine matrix that will be inverted.
        A peudo-inverse is used if `ndim_inter != ndim_in`

    Returns
    -------
    output_affine : (..., ndim_out[+1], ndim_in+1) tensor
        The returned matrix is homogeneous iff `a` is too.

    """
    return affine_matmul(a, affine_inv(b))


def affine_resize(affine, shape, factor, anchor='c'):
    """Update an affine matrix according to a resizing of the lattice.

    Notes
    -----
    This function is related to the `resize` function, which allows the
    user to choose between modes:
        * 'c' or 'centers': align centers
        * 'e' or 'edges':   align edges
        * 'f' or 'first':   align center of first voxel
        * 'l' or 'last':    align center of last voxel
    In cases 'c' and 'e', the volume shape is multiplied by the zoom
    factor (and eventually truncated), and two anchor points are used
    to determine the voxel size (neurite's behavior corresponds to 'c').
    In cases 'f' and 'l', a single anchor point is used so that the voxel
    size is exactly divided by the zoom factor. This case with an integer
    factor corresponds to subslicing the volume (e.g., vol[::f, ::f, ::f]).

    Parameters
    ----------
    affine : (..., ndim_out[+1], ndim_in+1) tensor
        Input affine matrix.
    shape : (ndim,) sequence[int]
        Input shape.
    factor : float or sequence[float]
        Resizing factor.
        * > 1 : larger image <-> smaller voxels
        * < 1 : smaller image <-> larger voxels
    anchor : {'centers', 'edges', 'first', 'last'} or list, default='centers'
        Anchor points.

    Returns
    -------
    affine : (..., ndim_out[+1], ndim_in+1) tensor
        Resized affine matrix.
    shape : (ndim,) tuple[int]
        Resized shape.

    """

    # read parameters
    affine = torch.as_tensor(affine)
    nb_dim = affine.shape[-1] - 1
    factor = utils.make_vector(factor, nb_dim).tolist()
    anchor = [a[0].lower() for a in py.make_list(anchor, nb_dim)]
    info = {'dtype': affine.dtype, 'device': affine.device}

    # compute output shape
    shape_out = [max(1, int(s * f)) for s, f in zip(shape, factor)]

    # compute shift and scale in each dimension
    shifts = []
    scales = []
    for anch, f, inshp, outshp in zip(anchor, factor, shape, shape_out):
        if inshp == 1 or outshp == 1:
            # anchor must be "edges"
            anch = 'e'
        if anch == 'c':
            shifts.append(0)
            scales.append((inshp - 1) / (outshp - 1))
        elif anch == 'e':
            shifts.append(0.5 * (inshp/outshp - 1))
            scales.append(inshp/outshp)
        elif anch == 'f':
            shifts.append(0)
            scales.append(1/f)
        elif anch == 'l':
            shifts.append((inshp - 1) - (outshp - 1) / f)
            scales.append(1/f)
        else:
            raise ValueError('Unknown anchor {}'.format(anch))
        if inshp == outshp == 1:
            scales[-1] = 1/f

    # build voxel-to-voxel transformation matrix
    lin = torch.diag(torch.as_tensor(scales, **info))
    trl = torch.as_tensor(shifts, **info)[..., None]
    trf = torch.cat((lin, trl), dim=1)

    # compose
    affine = affine_matmul(affine, as_euclidean(trf))
    return affine, tuple(shape_out)


def affine_sub(affine, shape, indices):
    """Update an affine matrix according to a sub-indexing of the lattice.

    Notes
    -----
    .. Only sub-indexing that *keep an homogeneous voxel size* are allowed.
       Therefore, indices must be `None` or of type `int`, `slice`, `ellipsis`.

    Parameters
    ----------
    affine : (..., ndim_out[+1], ndim_in+1) tensor
        Input affine matrix.
    shape : (ndim_in,) sequence[int]
        Input shape.
    indices : tuple[slice or ellipsis]
        Subscripting indices.

    Returns
    -------
    affine : (..., ndim_out[+1], ndim_new+1) tensor
        Updated affine matrix.
    shape : (ndim_new,) tuple[int]
        Updated shape.

    """
    def is_int(elem):
        if torch.is_tensor(elem):
            return elem.dtype in (torch.int32, torch.int64)
        elif isinstance(elem, int):
            return True
        else:
            return False

    def to_int(elem):
        if torch.is_tensor(elem):
            return elem.item()
        else:
            assert isinstance(elem, int)
            return elem

    # check types
    affine = torch.as_tensor(affine)
    nb_dim = affine.shape[-1] - 1
    info = {'dtype': affine.dtype, 'device': affine.device}
    if torch.is_tensor(shape):
        shape = shape.tolist()
    if len(shape) != nb_dim:
        raise ValueError('Expected shape of length {}. Got {}'
                         .format(nb_dim, len(shape)))
    if not isinstance(indices, tuple):
        raise TypeError('Indices should be a tuple.')
    indices = list(indices)

    # compute the number of input dimension that correspond to each index
    #   > slice index one dimension but eliipses index multiple dimension
    #     and their number must be computed.
    nb_dims_in = []
    ind_ellipsis = None
    for n_ind, ind in enumerate(indices):
        if isinstance(ind, slice):
            nb_dims_in.append(1)
        elif ind is Ellipsis:
            if ind_ellipsis is not None:
                raise ValueError('Cannot have more than one ellipsis.')
            ind_ellipsis = n_ind
            nb_dims_in.append(-1)
        elif is_int(ind):
            nb_dims_in.append(1)
        elif ind is None:
            nb_dims_in.append(0)
        else:
            raise TypeError('Indices should be None, integers, slices or '
                            'ellipses. Got {}.'.format(type(ind)))
    nb_known_dims = sum(nb_dims for nb_dims in nb_dims_in if nb_dims > 0)
    if ind_ellipsis is not None:
        nb_dims_in[ind_ellipsis] = max(0, nb_dim - nb_known_dims)

    # transform each index into a slice
    # note that we don't need to know "stop" to update the affine matrix
    nb_ind = 0
    indices0 = indices
    indices = []
    for d, ind in enumerate(indices0):
        if isinstance(ind, slice):
            start = ind.start
            step = ind.step
            step = 1 if step is None else step
            start = 0 if (start is None and step > 0) else \
                    shape[nb_ind] - 1 if (start is None and step < 0) else \
                    shape[nb_ind] + start if start < 0 else \
                    start
            indices.append(slice(start, None, step))
            nb_ind += 1
        elif ind is Ellipsis:
            for dd in range(nb_ind, nb_ind + nb_dims_in[d]):
                start = 0
                step = 1
                indices.append(slice(start, None, step))
                nb_ind += 1
        elif is_int(ind):
            indices.append(to_int(ind))
        elif ind is None:
            assert (ind is None), "Strange index of type {}".format(type(ind))
            indices.append(None)

    # Extract shift and scale in each dimension
    shifts = []
    scales = []
    slicer = []
    shape_out = []
    for d, ind in enumerate(indices):
        # translation + scale
        if isinstance(ind, slice):
            shifts.append(ind.start)
            scales.append(ind.step)
            shape_out.append(shape[d] // abs(ind.step))
            slicer.append(slice(None))
        elif isinstance(ind, int):
            scales.append(0)
            shifts.append(ind)
            slicer.append(0)
        else:
            slicer.append(None)
            assert (ind is None), "Strange index of type {}".format(type(ind))

    # build voxel-to-voxel transformation matrix
    lin = torch.diag(torch.as_tensor(scales, **info))
    if any(not isinstance(s, slice) for s in slicer):
        # drop/add columns
        lin = torch.unbind(lin, dim=-1)
        zero = torch.zeros(len(shifts), **info)
        new_lin = []
        for s in slicer:
            if isinstance(s, slice):
                col, *lin = lin
                new_lin.append(col)
            elif isinstance(s, int):
                col, *lin = lin
            elif s is None:
                new_lin.append(zero)
        lin = torch.stack(new_lin, dim=-1) if new_lin else []
    trl = torch.as_tensor(shifts, **info)[..., None]
    trf = torch.cat((lin, trl), dim=1) if len(lin) else trl

    # compose
    affine = affine_matmul(affine, as_euclidean(trf))
    return affine, tuple(shape_out)


def affine_pad(affine, shape, padsize, side=None):
    """Update an affine matrix according to a padding of the lattice.

    Parameters
    ----------
    affine : (..., ndim_out[+1], ndim_in+1) tensor
        Input affine matrix.
    shape : (ndim_in,) sequence[int]
        Input shape.
    padsize : sequence of int
        Padding size.
    side : {'left', 'right', 'both', None}, defualt=None
        Side to pad. If None, padding size for the left and right side
        should be provided in alternate order.

    Returns
    -------
    affine : (..., ndim_out[+1], ndim_new+1) tensor
        Updated affine matrix.
    shape : (ndim_new,) tuple[int]
        Updated shape.

    """

    dim = affine.shape[-1] - 1
    backend = utils.backend(affine)

    padsize = py.make_list(padsize, dim, crop=False)
    if side == 'left':
        padsize = [val for pair in zip(padsize, [0]*len(padsize))
                   for val in pair]
    elif side == 'right':
        padsize = [val for pair in zip([0]*len(padsize), padsize)
                   for val in pair]
    elif side == 'both':
        padsize = [val for pair in zip(padsize, padsize)
                   for val in pair]

    if len(padsize) != 2*dim:
        raise ValueError('Not enough padding values')
    padsize = list(zip(padsize[::2], padsize[1::2]))

    # Extract shift and scale in each dimension
    shifts = []
    shape_out = []
    for d, (low, up) in enumerate(padsize):
        # translation + scale
            shifts.append(-low)
            shape_out.append(shape[d] + low + up)

    # build voxel-to-voxel transformation matrix
    lin = torch.eye(dim, **backend)
    trl = torch.as_tensor(shifts, **backend)[..., None]
    trf = torch.cat((lin, trl), dim=1) if len(lin) else trl

    # compose
    affine = affine_matmul(affine, as_euclidean(trf))
    return affine, tuple(shape_out)


def affine_permute(affine, perm=None, shape=None):
    """Update an affine matrix according to a permutation of the lattice dims.

    Parameters
    ----------
    affine : (..., ndim_out[+1], ndim_in+1) tensor
        Input affine matrix.
    perm : sequence[int], optional
        Permutation of the lattice dimensions.
        By default, reverse dimension order.
    shape : (ndim_in,) sequence[int], optional
        Input shape.

    Returns
    -------
    affine : (..., ndim_out[+1], ndim_in+1) tensor
        Updated affine matrix.
    shape : (ndim_in,) tuple[int], optional
        Updated shape.
    """
    nb_dim = affine.shape[-1] - 1
    if perm is None:
        perm = list(range(nb_dim-1, -1, -1))
    if torch.is_tensor(perm):
        perm = perm.tolist()
    if len(perm) != nb_dim:
        raise ValueError('Expected perm to have {} elements. Got {}.'
                         .format(nb_dim, len(perm)))
    is_h = is_homogeneous(affine)
    affine = affine[..., :, perm + [-1]]
    affine = as_homogeneous(affine) if is_h else as_euclidean(affine)
    if shape is not None:
        shape = tuple(shape[p] for p in perm)
        return affine, shape
    else:
        return affine


def affine_transpose(affine, dim0, dim1, shape):
    """Update an affine matrix according to a transposition of the lattice.

    A transposition is a permutation that only impacts two dimensions.

    Parameters
    ----------
    affine : (..., ndim[+1], ndim+1) tensor
        Input affine matrix.
    dim0 : int
        Index of the first dimension
    dim1 : int
        Index of the second dimension
    shape : (ndim,) sequence[int], optional
        Input shape.

    Returns
    -------
    affine : (..., ndim[+1], ndim+1) tensor
        Updated affine matrix.
    shape : (ndim,) tuple[int], optional
        Updated shape.
    """
    affine = torch.as_tensor(affine)
    nb_dim = affine.shape[-1] - 1
    perm = list(range(nb_dim))
    perm[dim0] = dim1
    perm[dim1] = dim0
    return affine_permute(affine, perm, shape)


def affine_conv(affine, shape, kernel_size, stride=1, padding=0,
                output_padding=0, dilation=1, transposed=False):
    """Update an affine matrix according to a convolution of the lattice.

    Parameters
    ----------
    affine : (..., ndim[+1], ndim+1) tensor
        Input affine matrix.
    shape : (ndim,) sequence[int]
        Input shape.
    kernel_size : int or sequence[int]
        Kernel size
    stride : int or sequence[int], default=1
        Strides (= step size when moving the kernel)
    padding : int or sequence[int], default=0
        Amount of padding added to (both sides of) the input
    output_padding : int or sequence[int], default=0
        Additional size added to (the bottom/right) side of each
        dimension in the output shape. Only used if `transposed is True`.
    dilation : int or sequence[int], default=1
        Dilation (= step size between elements of the kernel)
    transposed : bool, default=False
        Transposed convolution.

    Returns
    -------
    affine : (..., ndim[+1], ndim+1) tensor
        Updated affine matrix.
    shape : (ndim,) tuple[int]
        Updated shape.

    """
    affine = torch.as_tensor(affine)
    info = {'dtype': affine.dtype, 'device': affine.device}
    ndim = affine.shape[-1] - 1
    if len(shape) != ndim:
        raise ValueError('Affine and shape not consistant. Found dim '
                         '{} and {}.'.format(ndim, len(shape)))
    kernel_size = py.make_list(kernel_size, ndim)
    stride = py.make_list(stride, ndim)
    padding = py.make_list(padding, ndim)
    output_padding = py.make_list(output_padding, ndim)
    dilation = py.make_list(dilation, ndim)

    # compute new shape and scale/offset that transform the
    # new lattice into the old lattice
    oshape = []
    scale = []
    offset = []
    for L, S, Pi, D, K, Po in zip(shape, stride, padding,
                                  dilation, kernel_size, output_padding):
        if K <= 0:
            K = S
        if Pi == 'auto':
            if K % 2 == 0:
                raise ValueError('Cannot compute automatic padding '
                                 'for even-sized kernels.')
            Pi = D * (K // 2)
        if transposed:
            oshape += [(L - 1) * S - 2 * Pi + D * (K - 1) + Po + 1]
            scale += [1/S]
            offset += [(Pi - (K-1)/2)/S]
        else:
            oshape += [math.floor((L + 2 * Pi - D * (K - 1) - 1) / S + 1)]
            scale += [S]
            offset += [(K-1)/2 - Pi]

    # build voxel-to-voxel transformation matrix
    lin = torch.diag(torch.as_tensor(scale, **info))
    trl = torch.as_tensor(offset, **info)[..., None]
    trf = torch.cat((lin, trl), dim=1)

    # compose
    affine = affine_matmul(affine, as_euclidean(trf))
    return affine, tuple(oshape)


def affine_default(shape, voxel_size=1., layout=None, center=0.,
                   dtype=None, device=None):
    """Generate an orientation matrix with the origin in the center of the FOV.

    Parameters
    ----------
    shape : sequence[int]
        Lattice shape.
    voxel_size : sequence[float], default=1
        Lattice voxel size
    layout : str or layout_like, default='RAS'
        Lattice layout (see `volume_layout`).
    center : sequence[float], default=0
        World-coordinate of the center of the field-of-view.
    dtype : dtype, optional
    device : device, optional

    Returns
    -------
    affine : (ndim+1, ndim+1) tensor
        Orientation matrix

    """
    backend = dict(dtype=dtype or utils.max_dtype(voxel_size, center),
                   device=device or utils.max_device(voxel_size, center))
    shape = utils.make_vector(shape)
    nb_dim = len(shape)
    voxel_size = utils.make_vector(voxel_size, nb_dim, **backend)
    center = utils.make_vector(center, nb_dim, **backend)
    shape = shape.to(**backend)

    # build layout
    if layout is None:
        layout = volume_layout(index=list(range(nb_dim)))
    else:
        layout = volume_layout(layout)
    layout = layout_matrix(layout, voxel_size=voxel_size, shape=shape,
                           dtype=dtype, device=device)

    # compute shift
    lin = layout[:nb_dim, :nb_dim]
    shift = center - linalg.matvec(lin, (shape-1)/2.)
    affine = torch.cat((lin, shift[:, None]), dim=1)

    return affine_make_homogeneous(as_euclidean(affine))


def affine_modify(affine, shape, voxel_size=None, layout=None, center=None):
    """Modifies features from an affine (voxel size, center, layout).

    Parameters
    ----------
    affine : (ndim+1, ndim+1) tensor_like
        Orientation matrix.
    shape : sequence[int]
        Lattice shape.
    voxel_size : sequence[float], optional
        Lattice voxel size
    layout : str or layout_like, optional
        Lattice layout (see `volume_layout`).
    center : sequence[float], optional
        World-coordinate of the center of the field-of-view.

    Returns
    -------
    affine : (ndim+1, ndim+1) tensor
        Orientation matrix

    """
    affine = torch.as_tensor(affine).clone()
    backend = utils.backend(affine)
    nb_dim = affine.shape[-1] - 1
    shape = utils.make_vector(shape, nb_dim, **backend)

    if center is None:
        # preserve center
        center = (shape - 1) * 0.5
        center = affine_matvec(affine, center)

    # set correct layout
    if layout is not None:
        affine, _ = affine_reorient(affine, shape.tolist(), layout)

    # set correct voxel size
    if voxel_size is not None:
        voxel_size = utils.make_vector(voxel_size, nb_dim, **backend)
        old_voxel_size = _voxel_size(affine)
        affine[:-1, :-1] *= voxel_size[None] / old_voxel_size[None]

    # set correct center
    center = utils.make_vector(center, nb_dim, **backend)
    vox_center = (shape - 1) * 0.5
    old_center = affine_matvec(affine, vox_center)
    affine[:-1, -1] += center - old_center

    return affine


def affine_reorient(mat, shape_or_tensor=None, layout=None):
    """Reorient an affine matrix / a tensor to match a target layout.

    Parameters
    ----------
    mat : (dim[+1], dim+1) tensor_like
        Orientation matrix
    shape_or_tensor : (dim,) sequence[int] or (..., *shape) tensor_like, optional
        Shape or Volume or MappedArray
    layout : layout_like, optional
        Target layout. Defaults to a directed layout (equivalent to 'RAS').

    Returns
    -------
    mat : (dim[+1], dim+1) tensor
        Reoriented orientation matrix
    shape_or_tensor : (dim,) tuple or (..., *permuted_shape) tensor, optional
        Reoriented shape or volume

    """
    # NOTE: some of the code is a bit weird (no use of tensor.dim(),
    # no torch.as_tensor()) so that we can work with MappedArray inputs
    # without having to import the MappedArray class. Methods `permute`
    # and `flip` are implemented in MappedArray so we're good.

    # parse inputs
    mat = torch.as_tensor(mat).clone()
    dim = mat.shape[-1] - 1
    shape = tensor = None
    if shape_or_tensor is not None:
        if not hasattr(shape_or_tensor, 'shape'):
            shape_or_tensor = torch.as_tensor(shape_or_tensor)
        if len(shape_or_tensor.shape) > 1:
            tensor = shape_or_tensor
            shape = tensor.shape[-dim:]
        else:
            shape = shape_or_tensor
            if torch.is_tensor(shape):
                shape = shape.tolist()[-dim:]
            shape = tuple(shape)

    # find current layout and target layout
    #   layouts are (dim, 2) tensors where
    #       - the first column stores indices of the axes in RAS space
    #         (and is therefore a permutation that transforms a RAS
    #          space into this layout)
    #       - the second column stores 0|1 values that indicate whether
    #         this axis (after permutation) is flipped.
    current_layout = affine_to_layout(mat)
    target_layout = volume_layout(dim if layout is None else layout)
    ras_to_current, current_flips = current_layout.unbind(-1)
    ras_to_target, target_flips = target_layout.unbind(-1)
    current_to_ras = utils.invert_permutation(ras_to_current)

    # compose flips (xor)
    current_flips = current_flips.bool()
    target_flips = target_flips.bool()
    ras_flips = current_flips[current_to_ras]
    target_flips = target_flips ^ ras_flips[ras_to_target]

    # compose permutations
    current_to_target = current_to_ras[ras_to_target]

    # apply permutation and flips
    if shape:
        mat, shape = affine_permute(mat, current_to_target, shape)
    else:
        mat = affine_permute(mat, current_to_target)
    index = tuple(slice(None, None, -1) if flip else slice(None)
                  for flip in target_flips)
    if shape:
        mat, _ = affine_sub(mat, shape, index)
    else:
        for d, flip in enumerate(target_flips):
            if flip:
                mat[:, d].neg_()

    if tensor is not None:
        is_numpy = not hasattr(tensor, 'permute')
        # we need to append stuff to take into account batch dimensions
        nb_dim_left = len(tensor.shape) - len(index)
        current_to_target = current_to_target + nb_dim_left
        current_to_target = list(range(nb_dim_left)) + current_to_target.tolist()
        if is_numpy:
            import numpy as np
            tensor = np.transpose(current_to_target)
        else:
            tensor = tensor.permute(current_to_target)
        dim_flip = [nb_dim_left + d for d, idx in enumerate(index) 
                    if idx.step == -1]
        if dim_flip:
            if is_numpy:
                tensor = np.flip(tensor, dim_flip)
            else:
                tensor = tensor.flip(dim_flip)
        return mat, tensor
    else:
        return (mat, tuple(shape)) if shape else mat


def compute_fov(mat, affines, shapes, pad=0, pad_unit='%'):
    """Compute the bounding box of spaces when projected in a target space.

    Parameters
    ----------
    mat : (D+1, D+1) tensor_like
        Output orientation matrix (up to a shift)
    affines : (N, D+1, D+1), tensor_like
        Input orientation matrices
    shapes : (N, D) tensor_like[int]
        Input shapes
    pad : [sequence of] float, default=0
        Amount of padding (or cropping) to add to the bounding box.
    pad_unit : [sequence of] {'mm', '%'}, default='%'
        Unit of the padding/cropping.

    Returns
    -------
    mn : (D,) tensor
        Minimum coordinates, in voxels (without floor/ceil)
    mx : (D,) tensor
        Maximum coordinates, in voxels (without floor/ceil)

    """
    mat = utils.as_tensor(mat)
    backend = dict(device=mat.device, dtype=mat.dtype)
    affines = utils.as_tensor(affines, **backend)
    shapes = utils.as_tensor(shapes, **backend)
    affines.reshape([-1, *affines.shape[-2:]])
    shapes.reshape([-1, shapes.shape[-1]])
    shapes = shapes.expand([len(affines), shapes.shape[-1]])
    dim = mat.shape[-1] - 1

    mn = torch.full([dim], float('inf'), **backend)
    mx = torch.full([dim], -float('inf'), **backend)
    for a_mat, a_shape in zip(affines, shapes):
        corners = itertools.product([False, True], r=dim)
        corners = [[a_shape[i] if top else 1 for i, top in enumerate(c)] + [1]
                   for c in corners]
        corners = torch.as_tensor(corners, **backend).T
        M = linalg.lmdiv(mat, a_mat)
        corners = torch.matmul(M[:dim, :], corners)
        mx = torch.max(mx, torch.max(corners, dim=1)[0])
        mn = torch.min(mn, torch.min(corners, dim=1)[0])
    if pad is None:
        pad = 0
    pad = utils.make_vector(pad, dim, **backend)
    if pad_unit == '%':
        pad = pad / 100.
        pad = pad * (mx - mn) / 2.
    mx = mx + pad
    mn = mn - pad
    return mn, mx