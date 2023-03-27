from matplotlib import pyplot as plt

if __name__ == '__main__':
    fn = 'out_1.csv'

    accs = []
    macss = []
    with open(fn, 'r') as f:
        lns = f.readlines()
        for ln in lns:
            ln = ln.strip()
            res = ln.split(', ')
            macs, acc = int(res[0]), float(res[1])
            accs.append(acc)
            macss.append(macs)

    fn = 'out_2.csv'

    with open(fn, 'r') as f:
        lns = f.readlines()
        for i, ln in enumerate(lns):
            if i < 3:
                continue
            ln = ln.strip()
            res = ln.split(', ')
            macs, acc = int(res[0]), float(res[1])
            accs.append(acc)
            macss.append(macs)

    base_score = [macss[0]], [accs[0]]
    max_score = [macss[1]], [accs[1]]
    min_score = [macss[2]], [accs[2]]
    plt.scatter(macss[3:], accs[3:], marker='.', alpha=0.3, label='Random')
    plt.scatter(base_score[0], base_score[1], marker='.', alpha=0.3, label='Base')
    plt.scatter(min_score[0], min_score[1], marker='.', alpha=0.3, label='Min')
    plt.scatter(max_score[0], max_score[1], marker='.', alpha=0.3, label='Max')

    plt.title('BNAS R50 CIFAR10 - {} subnets\nWith Reset BN'.format(len(macss)))
    plt.legend()
    plt.xlabel('MACCs')
    plt.ylabel('Top1 [%]')
    plt.savefig('out.png')
