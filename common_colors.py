#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2025 dlilien <dlilien@IU-FL419X7XRV>
#
# Distributed under terms of the MIT license.

"""

"""

import matplotlib.colors as mc
import colorsys

# Source - https://stackoverflow.com/a
# Posted by Ian Hincks, modified by community. See post 'Timeline' for change history
# Retrieved 2025-12-15, License - CC BY-SA 4.0
def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string.

    Exceed 1 to darken.
    """
    c = colorsys.rgb_to_hls(*mc.to_rgb(color))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


ncolors = ["#E69F00", "#56B4E9", "#009E73", "#F0E442"]


color_dict = {
    "identical": {
        1.8: "C2",
        3.0: "C0",
        3.5: "C5",
        4.0: "C9",
    },
    "standard": {
        1.8: "C1",
        3.0: "C3",
        3.5: "C6",
        4.0: "C8",
    },
    "true": {3.0: "k"},
}

color_dict_T = {init: {n: {-20: lighten_color(c, 0.5),
                           -12: lighten_color(c, 0.5),
                           -10: c,
                           -8: lighten_color(c, 1.3),
                           -5: lighten_color(c, 1.3),
                           } for n, c in color_dict[init].items()} for init in color_dict}

color_dict_lightdark = {init: {n: {-20: c,
                                  -12: c,
                                  -10: c,
                                  -8: c,
                                   -5: c,
                                  } for n, c in color_dict[init].items()} for init in color_dict}

marker_ec_dict = {init: {n: {-20: "k",
                             -12: "k",
                             -10: c,
                             -8: "0.3",
                             -5: "0.3",
                             } for n, c in color_dict[init].items()} for init in color_dict}



linewdiths = {-20: 0.5,
              -12: 0.5,
              -10: 2,
              -8: 3,
              -5: 3
              }
