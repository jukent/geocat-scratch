################################################################
    #  Globals for random numbers.

    global dfran_iseq
    dfran_iseq = seed

    #  Check arguments.

    try:
        alen = len(dims)
    except:
        print(
            "generate_2d_array: first argument must be a list, tuple, or array having two elements specifying the dimensions of the output array."
        )
        return None
    if (alen != 2):
        print(
            "generate_2d_array: first argument must have two elements specifying the dimensions of the output array."
        )
        return None
    if (int(dims[0]) <= 1 and int(dims[1]) <= 1):
        print("generate_2d_array: array must have at least two elements.")
        return None
    if (num_low < 1):
        print(
            "generate_2d_array: number of lows must be at least 1 - defaulting to 1."
        )
        num_low = 1
    if (num_low > 25):
        print(
            "generate_2d_array: number of lows must be at most 25 - defaulting to 25."
        )
        num_high = 25
    if (num_high < 1):
        print(
            "generate_2d_array: number of highs must be at least 1 - defaulting to 1."
        )
        num_high = 1
    if (num_high > 25):
        print(
            "generate_2d_array: number of highs must be at most 25 - defaulting to 25."
        )
        num_high = 25
    if (seed > 100 or seed < 0):
        print(
            "generate_2d_array: seed must be in the interval [0,100] - seed set to 0."
        )
        seed = 0
    if not lows_at is None:
        if (len(lows_at) != num_low):
            print(
                "generate_2d_array: the list of positions for the lows must be the same size as num_low."
            )
    if not highs_at is None:
        if (len(highs_at) != num_high):
            print(
                "generate_2d_array: the list of positions for the highs must be the same size as num_high."
            )


#  Dims are reversed in order to get the same results as the NCL function.

    nx = int(dims[1])
    ny = int(dims[0])
    out_array = np.zeros([nx, ny], 'f')
    tmp_array = np.zeros([3, 51], 'f')
    fovm = 9. / float(nx)
    fovn = 9. / float(ny)
    nlow = max(1, min(25, num_low))
    nhgh = max(1, min(25, num_high))
    ncnt = nlow + nhgh

    for k in range(num_low):
        if not lows_at is None:
            tmp_array[0,
                      k] = float(lows_at[k][1])  # lows at specified locations.
            tmp_array[1, k] = float(lows_at[k][0])
            tmp_array[2, k] = -1.
        else:
            tmp_array[0, k] = 1. + (float(nx) -
                                    1.) * _dfran()  # lows at random locations.
            tmp_array[1, k] = 1. + (float(ny) -
                                    1.) * _dfran()  # lows at random locations.
            tmp_array[2, k] = -1.
    for k in range(num_low, num_low + num_high):
        if not highs_at is None:
            tmp_array[0, k] = float(highs_at[k - num_low][1])  # highs locations
            tmp_array[1, k] = float(highs_at[k - num_low][0])  # highs locations
            tmp_array[2, k] = 1.
        else:
            tmp_array[0, k] = 1. + (float(nx) -
                                    1.) * _dfran()  # highs at random locations.
            tmp_array[1, k] = 1. + (float(ny) -
                                    1.) * _dfran()  # highs at random locations.
            tmp_array[2, k] = 1.

    dmin = 1.e+36
    dmax = -1.e+36
    midpt = 0.5 * (minv + maxv)
    for j in range(ny):
        for i in range(nx):
            out_array[i, j] = midpt
            for k in range(ncnt):
                tempi = fovm * (float(i + 1) - tmp_array[0, k])
                tempj = fovn * (float(j + 1) - tmp_array[1, k])
                temp = -(tempi * tempi + tempj * tempj)
                if (temp >= -20.):
                    out_array[i,j] = out_array[i,j] +    \
                       0.5*(maxv - minv)*tmp_array[2,k]*math.exp(temp)
            dmin = min(dmin, out_array[i, j])
            dmax = max(dmax, out_array[i, j])

    out_array = (((out_array - dmin) / (dmax - dmin)) * (maxv - minv)) + minv

    del tmp_array

    return np.transpose(out_array, [1, 0])


def _get_double(obj, name):
    return (NhlGetDouble(_int_id(obj), name))


def _get_double_array(obj, name):
    return (NhlGetDoubleArray(_int_id(obj), name))


# Create dummy data
nx = 100
ny = 100
data1 = generate_2d_array((ny, nx), 10, 10, -19., 16., 0)
data2 = generate_2d_array((ny, nx), 10, 10, -28., 15., 1)
data3 = generate_2d_array((ny, nx), 10, 10, -25., 18., 2)
