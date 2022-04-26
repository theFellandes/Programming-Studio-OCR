increment = 1
resistances = [22_000, 10_000, 2200]
changed = [3300, 4700, 9000, 10_000, 33000, 47000]
for i in changed:
    # r serial calculates the resistance that is serial
    r_serial = resistances[2] + i
    # parallel connects with 10k ohm
    r_parr = (1 / resistances[1]) + (1 / r_serial)
    r_parr = 1 / r_parr
    # equivalent resistance
    r_eq = resistances[0] + r_parr
    # total current
    curr = 12 / r_eq
    # parallel voltage
    v_parr = r_parr * curr
    # one branch
    i_serial = v_parr / r_serial
    # voltage
    v = i_serial * i
    # current
    current = v / i
    # power
    p = v * current
    print('# ' + str(increment))
    increment += 1
    print("v: " + str(v) + " V")
    print("i: " + str(current) + " A")
    print("P: " + str(p) + " W")
    print("-----------------------")
