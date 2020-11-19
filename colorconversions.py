def hex_to_rgb(value, strip_alpha=True):
    """
    Converts hex to rgb colours
    value: string of 6 characters representing a hex colour.
    Returns: list length 3 of RGB values

    # Color Conversion code taken from:
    https://gist.github.com/KerryHalupka/df046b971136152b526ffd4be2872b9d

    """

    value = value.strip("#")  # removes hash symbol if present

    # strip alpha channel
    if strip_alpha and len(value) == 8:
        value = value[:-2]

    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


def rgb_to_dec(value):
    """
    Converts rgb to decimal colours (i.e. divides each value by 256)
    value: list (length 3) of RGB values
    Returns: list (length 3) of decimal values
    """

    return [v / 256 for v in value]


def create_cmap_from_czi_channelcolors(channelcolors, numchannels=1):

    cmaps = []

    if channelcolors is None:
        for ch in range(numchannels):
            cmaps.append(cm.gray)

    if channelcolors is not None:
        for cc in channelcolors:

            values = rgb_to_dec(hex_to_rgb(cc))
            cmaps.append(values)

    return cmaps
