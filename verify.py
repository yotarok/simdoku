import sys

def check_line(l):
    return len(l) == len(set(l))

def verify(s):
    for r in range(9):
        d = s[r*9:r*9+9]
        if not check_line(d):
            print("Error: {}: row = {}: {}".format(s, r, d))
    for c in range(9):
        d = s[c::9]
        if not check_line(d):
            print("Error: {}: col = {}: {}".format(s, c, d))
    for br in range(3):
        for bc in range(3):
            d = ''
            for off in range(3):
                idx = (br * 3 + off) * 9 + bc * 3
                d += s[idx:idx + 3]
            if not check_line(d):
                print("Error: {}: block = {},{}: {}".format(s, br, bc, d))

for l in sys.stdin:
    verify(l.strip())
