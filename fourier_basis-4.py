fig, axes = plt.subplots(1, 2, figsize=(10, 5))
dftp.show_array(axes[0], dftp.scale_array(C))
axes[0].set_title("$\mathbf{C}$")
# <...>
dftp.show_array(axes[1], dftp.scale_array(S))
axes[1].set_title("$\mathbf{S}$")
# <...>
