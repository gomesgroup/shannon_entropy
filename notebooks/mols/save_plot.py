def save_bond_angle(mol_name, angle_min, angle_max, num_points, idx_1=2, idx_2=1, idx_3=3, basis='sto-3g', xc='B3LYP'):
    x = np.linspace(angle_min, angle_max, num=num_points)
    y = np.zeros(len(x))

    for i in range(len(x)):
        print(i)
        mol_ob = adjust_bond_angle(mol_name, idx_1, idx_2, idx_3, x[i])
        y[i] = calc_coper(mol_ob, basis=basis, xc=xc)

    # Export data to CSV
    data = np.column_stack((x, y))
    np.savetxt(str.format("data/bond_angle/{mol}_bondangle_{min}_{max}_{num}.csv", 
               mol=mol_name, min=angle_min, max=angle_max, num=num_points),
               data, delimiter=',', header='angle,coper', comments='')