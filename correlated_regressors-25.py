B_boths_o = npl.pinv(X_both_o).dot(Ys)
# Distribution of parameter for hrf1 in orth model
plt.hist(B_boths_o[0], bins=50)
# (...)
print(np.mean(B_boths_o[0]), np.std(B_boths_o[0]))
# 1.68134012906 1.47669405469
