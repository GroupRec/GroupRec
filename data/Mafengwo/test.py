
if __name__ == "__main__":
    with open('./userRatingTest.txt') as f1:
        l_1 = f1.readlines()
    with open('./userRatingNegative.txt') as f2:
        l_2 = f2.readlines()

    u1 = []
    for lines in l_1:
        u1.append(lines.split(' ')[0])
    u2 = []
    for lines in l_2:
        # print(lines)
        u2.append(lines.split(' ')[0].split(',')[0][1:])
    cnt = 0
    for i in range(len(u1)):
        if u1[i] != u2[i]:
            cnt = i
            break
    print(cnt)
