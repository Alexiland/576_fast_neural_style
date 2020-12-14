from matplotlib import pyplot as plt

# name: [loss, flops]
candy_data = {
    "0/0": [440607.369709, 157859840],
    "4/0": [440488.207610, 88961024],
    # "4/4": [571748.860582, 20062208],
    "6/6": [450690.425261, 29904896],
    "8/0": [441463.535750, 98803712],
    "8/8": [441941.766487, 39747584],
    "16/0": [441395.089719, 118489088],
    "16/16": [441345.598864, 79118336]
}

mosaic_dta = {
    "0/0": [1265839.017366, 157859840],
    "4/0": [1264640.815575, 88961024],
    # "4/4": [1677506.466144, 20062208], #x
    "6/6": [1284549.008266, 29904896],
    "8/0": [1265786.818943, 98803712],
    "8/8": [1265786.818943, 39747584], #x
    "16/0": [1263650.480286, 118489088],
    "16/16": [1264388.290647, 79118336]
}

fig, ax = plt.subplots()

loss = [item[0] for item in candy_data.values()]
flops = [item[1] for item in candy_data.values()]

ax.scatter(flops, loss)

for i, txt in enumerate(candy_data.keys()):
    ax.annotate(txt, (flops[i], loss[i]))
# plt.title("Quantized Transform Network Loss vs. FLOPs")
plt.xlabel("FLOPs")
plt.ylabel("Total Loss")
# plt.show()
plt.savefig("lossvsflops")