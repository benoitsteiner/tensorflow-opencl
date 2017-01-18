# Description:
#   A library for decoding and encoding GIF images

licenses(["notice"])  # MIT

exports_files(["COPYING"])

cc_library(
    name = "gif",
    srcs = [
        "lib/dgif_lib.c",
        "lib/egif_lib.c",
        "lib/gif_err.c",
        "lib/gif_font.c",
        "lib/gif_hash.c",
        "lib/gif_hash.h",
        "lib/gif_lib_private.h",
        "lib/gifalloc.c",
        "lib/openbsd-reallocarray.c",
        "lib/quantize.c",
    ],
    hdrs = ["lib/gif_lib.h"],
    includes = ["lib/."],
    visibility = ["//visibility:public"],
    deps = select({
        ":windows": [":windows_polyfill"],
        "//conditions:default": [],
    }),
)

cc_library(
    name = "windows_polyfill",
    hdrs = ["windows/unistd.h"],
    includes = ["windows"],
)

genrule(
    name = "windows_unistd_h",
    outs = ["windows/unistd.h"],
    cmd = "touch $@",
)

config_setting(
    name = "windows",
    values = {"cpu": "x64_windows_msvc"},
)
