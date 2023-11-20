import torch


def compare(expected: torch.Tensor, taken: torch.Tensor, enable_dump:bool = True, num_vals:int = 20):
    if expected.dtype != taken.dtype:
        print(f"Dtypes are different: expected - {expected.dtype}, taken - {taken.dtype}")
        return False
    if expected.shape != taken.shape:
        print(f"Shapes are different: expected - {expected.shape}, taken - {taken.shape}")
        return False
    if torch.is_floating_point(expected) and not torch.allclose(expected, taken): 
        mask = torch.isclose(expected, taken)
        coord_sz = 2 * len(expected.shape) + 1
        coord_sz = coord_sz - len("coord")
        header = "{} {:>14} {:>14} dtype - {}".format(coord_sz*" " + "coord", "expected", "taken", expected.dtype)
        print(header)
        mism = torch.where(~mask)
        print(len(mism[0]), " values are different.")
        is_dump_needed = enable_dump and (len(mism[0]) > num_vals)
        dump_name = 'mismatch.txt'
        if is_dump_needed:
            f = open(dump_name, 'w+')
            f.write(header + "\n")
        for n, it in enumerate(mism):
            line = f"{it.numpy()} {expected[*it].item():.12f} {taken[*it].item():.12f}" 
            if is_dump_needed:
                f.write(line + "\n")
            if n < num_vals:
                print(line)
        if is_dump_needed:
            print("Full diff is stored in ", dump_name)
            f.close()
        return False
    if not torch.is_floating_point(expected) and not torch.eq(expected, taken):
        mask = expected.eq(taken)
        coord_sz = 2 * len(expected.shape) + 1
        coord_sz = coord_sz - len("coord")
        header = "{} {:>14} {:>14} dtype - {}".format(coord_sz*" " + "coord", "expected", "taken", expected.dtype)
        print(header)
        mism = torch.where(~mask)
        print(len(mism[0]), " values are different.")
        is_dump_needed = enable_dump and (len(mism[0]) > num_vals)
        dump_name = 'mismatch.txt'
        if is_dump_needed:
            f = open(dump_name, 'w+')
            f.write(header + "\n")
        for n, it in enumerate(mism):
            line = f"{it.numpy()} {expected[*it].item():14d} {taken[*it].item():.14d}" 
            if is_dump_needed:
                f.write(line + "\n")
            if n < num_vals:
                print(line)
        if is_dump_needed:
            print("Full diff is stored in ", dump_name)
            f.close()
    print("All matched!")
    return True