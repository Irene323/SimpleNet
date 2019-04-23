# # use a list to load all the data (better into match & nonmatch) then use 2 for loop
# match = []
# nonmatch = []
# with open('D:\\dataset\\liberty\\m50_500000_500000_0.txt', 'r') as f:
#     for l in f:
#         a = l.split(' ')
#         if (a[1] != a[4]):
#             nonmatch.append([a[0],a[3]])
#         else:
#             match.append([a[0],a[3]])
#
# with open('merge_loop.txt', 'w') as merge:
#     for n in nonmatch:
#         for m in match:
#             if n[0] == m[0]:
#                 merge.write(m[0] + ' ' + m[1] + ' ' + n[1] + '\n')
#             elif n[0] == m[1]:
#                 merge.write(m[1] + ' ' + m[0] + ' ' + n[1] + '\n')
#             elif n[1] == m[0]:
#                 merge.write(m[0] + ' ' + m[1] + ' ' + n[0] + '\n')
#             elif n[1] == m[1]:
#                 merge.write(m[1] + ' ' + m[0] + ' ' + n[0] + '\n')


# with dict
match = {}
nmatch = {}
with open('D:\\dataset\\liberty\\m50_200000_200000_0.txt', 'r') as f:
    for l in f:
        a = l.split(' ')
        if a[1] == a[4]:
            if not a[0] in match.keys():
                match[a[0]] = []
            match[a[0]].append(a[3])
        else:
            if not a[0] in nmatch.keys():
                nmatch[a[0]] = []
            nmatch[a[0]].append(a[3])
            if not a[3] in nmatch.keys():
                nmatch[a[3]] = []
            nmatch[a[3]].append(a[0])
#
# print(match)
# print(nmatch)

with open('../resources/merge_200000.txt', 'w') as f:
    for k in match.keys():
        if not k in nmatch.keys():
            continue
        for m in match[k]:
            for n in nmatch[k]:
                f.write('{} {} {}\n'.format(k, m, n))
