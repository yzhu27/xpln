
import math
import re
import sys
import csv


the = {}
help = """
xpln: multi-goal semi-supervised explanati 
(c) 2023 Tim Menzies <timm@ieee.org> BSD-2
  
USAGE: lua xpln.lua [OPTIONS] [-g ACTIONS]
  
OPTIONS:
  -b  --bins    initial number of bins       = 16
  -c  --cliffs  cliff's delta threshold      = .147
  -d  --d       different is over sd*d       = .35
  -f  --file    data file                    = ../etc/data/auto93.csv
  -F  --Far     distance to distant          = .95
  -g  --go      start-up action              = nothing
  -h  --help    show help                    = false
  -H  --Halves  search space for clustering  = 512
  -m  --min     size of smallest cluster     = .5
  -M  --Max     numbers                      = 512
  -p  --p       dist coefficient             = 2
  -r  --rest    how many of rest to sample   = 4
  -R  --Reuse   child splits reuse a parent pole = true
  -s  --seed    random number seed           = 937162211
"""

# Create a `SYM` to summarize a stream of symbols.


def SYM(n=0, s=""):
    return {"at": n,
            "txt": s,
            "n": 0,
            "mode": None,
            "most": 0,
            "isSym": True,
            "has": {}
            }

# Create a `NUM` to summarize a stream of numbers.


def NUM(n=0, s=""):
    return {"at": n,
            "txt": s,
            "n": 0,
            "hi": -math.inf,
            "lo": math.inf,
            "ok": True,
            "isSym": False,
            "has": {},
            "w": -1 if (s or "").endswith("-") else 1}

# Create a `COL` to summarize a stream of data for a column.


def COL(n, s):
    col = NUM(n, s) if s.istitle() else SYM(n, s)
    col["isIgnored"] = col["txt"].endswith("X")
    col["isKlass"] = col["txt"].endswith("!")

    col["isGoal"] = col['txt'].endswith(('!', '+', '-'))
    return col

# Create a set of `NUM`s or `SYM`s columns.


def COLS(ss):
    cols = {"names": ss, "all": {}, "x": {}, "y": {}}
    for n, s in ss.items():
        # use push() here, defined in later code
        # col = cols['all'].update(COL(n, s))
        col = push(cols['all'], COL(n, s))

        if not col['isIgnored']:
            if col['isKlass']:
                cols['klass'] = col
            if col['isGoal']:
                # cols['y'].update(col)
                push(cols['y'], col)
            else:
                # cols['x'].update(col)
                push(cols['x'], col)
    return cols

# Create a RANGE  that tracks the y dependent values seen in
# the range `lo` to `hi` some independent variable in column number `at` whose name is `txt`.
# Note that the way this is used (in the `bins` function, below)
# for  symbolic columns, `lo` is always the same as `hi`.


def RANGE(at, txt, lo, hi=None):
    return {"at": at,
            "txt": txt,
            "lo": lo,
            "hi": hi if lo is None else lo,
            "y": SYM()}

# Create a `DATA` to contain `rows`, summarized in `cols`.


# Create a  RULE that groups `ranges` by their column id. 
# Each group is a disjunction of its contents (and
# sets of groups are conjunctions).
def RULE(ranges , maxSize):
    t = {}
    for _ , range in ranges.items():
        if range['txt'] not in t: t[range['txt']]={}
        push(t[range['txt']], {'lo':range['lo'],'hi':range['hi'],'at':range['at']})
    return prune(t, maxSize)

def prune(rule, maxSize):
    n = 0
    for txt,ranges in rule.items():
        n += 1
        if len(ranges) == maxSize[txt]:
            n += 1
            rule[txt] = {}
    if n > 0:
        return rule

class DATA:
    def __init__(self):
        self.rows = {}
        self.cols = None

    # Create a new DATA by reading csv file whose first row
    # are the comma-separate names processed by `COLS` (above).
    # into a new `DATA`. Every other row is stored in the DATA by
    # calling the
    # `row` function (defined below).

    # not sure what is t and csv()
    def read(sfile):
        data = DATA()
        # not sure csv() is correct or not

        def fun(x):
            row(data, x)
        Csv(sfile, fun)
        return data

    # Create a new DATA with the same columns as  `data`. Optionally, load up the new
    # DATA with the rows inside `ts`.
    def clone(data, ts=None):
        data1 = row(DATA(), data.cols["names"])
        if ts:
            for _, t in ts.items():
                row(data1, t)
        return data1

# Update `data` with  row `t`. If `data.cols`
# does not exist, the use `t` to create `data.cols`.
# Otherwise, add `t` to `data.rows` and update the summaries in `data.cols`.
# To avoid updating skipped columns, we only iterate
# over `cols.x` and `cols.y`.


def row(data, t):
    if data.cols:
        # data["rows"].update(t)
        push(data.rows, t)
        for _, col in data.cols["x"].items():

         # not sure what is add()
            add(col, t[col["at"]])
        for _, col in data.cols["y"].items():
            # oo(cols)

            # not sure what is add()
            add(col, t[col["at"]])
    else:
        data.cols = COLS(t)
    return data

# Update one COL with `x` (values from one cells of one row).
# Used  by (e.g.) the `row` and `adds` function.
# `SYM`s just increment a symbol counts.
# `NUM`s store `x` in a finite sized cache. When it
# fills to more than `the.Max`, then at probability
# `the.Max/col.n` replace any existing item
# (selected at random). If anything is added, the list
# may not longer be sorted so set `col.ok=false`.


def add(col, x, n=None):
    global the
    if x != "?":
        n = n or 1
        col['n'] += n
        if col['isSym']:
            col['has'][x] = n + (col['has'].get(x) or 0)
            if col['has'][x] > col['most']:
                col['most'], col['mode'] = col['has'][x], x
        else:
            col['lo'], col['hi'] = min(x, col['lo']), max(x, col['hi'])

            # all_ is all in lua
            # all_ and pos are local
            all_, pos = len(col['has']), None
            if all_ < the['Max']:
                pos = all_ + 1
            # rand() is defined in later code
            elif rand() < the['Max'] / col['n']:
                pos = rint(1, all_)
            if pos is not None:
                col['has'][pos] = x
                col['ok'] = False

# Update a COL with multiple items from `t`. This is useful when `col` is being
# used outside of some DATA.


def adds(col, t):
    for x in t or {}:
        add(col, x)
    return col

# Update a RANGE to cover `x` and `y`


def extend(range_, n, s):
    range_['lo'] = min(n, range_['lo'])
    range_['hi'] = max(n, range_['hi'])
    add(range_['y'], s)

# A query that returns contents of a column. If `col` is a `NUM` with
# unsorted contents, then sort before return the contents.
# Called by (e.g.) the `mid` and `div` functions.


def has(col):
    if not col['isSym'] and not col['ok']:
        col['has'] = sort(col['has'])
    col['ok'] = True
    return col['has']

# A query that returns a `cols`'s central tendency
# (mode for `SYM`s and median for `NUM`s). Called by (e.g.) the `stats` function.


def mid(col):
    if col['isSym'] and col['mode']:
        return col['mode']
    else:
        # has_ = has(col)
        # return per(has_, 0.5)
        return per(has(col), 0.5)

# A query that returns a `col`'s deviation from central tendency
# (entropy for `SYM`s and standard deviation for `NUM`s)..


def div(col):
    if col['isSym']:
        e = 0
        for _, n in col['has'].items():
            e -= n / col['n'] * math.log(n / col['n'], 2)
        return e
    else:
        has_ = has(col)
        return (per(has_, 0.9) - per(has_, 0.1)) / 2.58

# A query that returns `mid` or `div` of `cols` (defaults to `data.cols.y`).
def stats(data, fun=None, cols=None, nPlaces=None):
    cols = cols or data.cols['y']
    # # not sure follow code is correct or not
    # # rnd() is defined in later code
    # def statsfun(k,col):
    #     return rnd(fun or mid)
    # tmp = {k: (rnd((fun or mid)(col), nPlaces), col['txt']) for k, col in cols.items()}
    # tmp['N'] = len(data.rows)
    # return tmp, {k: mid(col) for k, col in cols.items()}

    def funn(k, col):
        if fun == 'div':
            return rnd(div(col), nPlaces)
        else:
            return rnd(mid(col), nPlaces)
    u = {}
    for i in range(len(cols)):
        k = cols[i]['txt']
        u[k] = funn(k, cols[i])
    res = {}
    for k in sorted(u.keys()):
        res[k] = u[k]
    res['N'] = len(data.rows)
    return res

# A query that normalizes `n` 0..1. Called by (e.g.) the `dist` function.


def norm(num, n):
    # what is 'x' here for???

    return (n - num["lo"])/(num["hi"]-num["lo"] + 1/math.inf)


def value(has, nB, nR, sGoal:str):
    nB=nB or 1
    nR=nR or 1
    sGoal = sGoal or True
    b, r = 0, 0
    for x, n in has.items():
        if x == sGoal:
            b = b + n
        else:
            r = r + n
    b, r = b/(nB+1/float('inf')), r/(nR+1/float('inf'))
    return b**2/(b+r)


def dist(data, t1, t2, cols):
    def dist1(col, x, y):
        if x == '?' and y == '?':
            return 1
        if col['isSym']:
            return 0 if x == y else 1
        else:
            x, y = norm(col, x), norm(col, y)
            if x == '?':
                x = 1 if y < 0.5 else 1
            if y == '?':
                y = 1 if x < 0.5 else 1
            return abs(x - y)
    d, n = 0, 1/float('inf')
    for _, col in (cols or data.cols['x']).items():
        n += 1
        d += dist1(col, t1[col['at']], t2[col['at']]) ** the['p']
    return (d/n)**(1/the['p'])


def better(data, row1, row2):
    s1 = 0
    s2 = 0
    ys = data.cols['y']
    for _, col in ys.items():
        x = norm(col, row1[col['at']])
        y = norm(col, row2[col['at']])
        s1 -= math.exp(col['w'] * (x - y) / len(ys))
        s2 -= math.exp(col['w'] * (y - x) / len(ys))
    return (s1 / len(ys)) - (s2 / len(ys))

def betters(data,  n):
    def fun(r1 , r2):
        return better(data,r1,r2)
    #tmp = sorted(data.rows.values(), key=fun) 
    tmp = data.rows
    return  n and slice(tmp,1,n), slice(tmp,n+1,None) or tmp

def half(data, rows=None, cols=None, above=None):
    left, right = {}, {}
    
    def gap(r1, r2):
        return dist(data, r1, r2, cols)

    def cos(a, b, c):
        if c != 0:
            return (a**2 + c**2 - b**2)/(2*c)
        else:
            return math.inf

    def proj(r):
        return {'row': r, 'x': cos(gap(r, A), gap(r, B), c)}
    rows = rows or data.rows
    some = many(rows, the['Halves'])
    A = (the['Reuse'] and above) or any(some)

    def fun(r):
        return {'row': r, 'd': gap(r, A)}
    tmp = sort2(map(some, fun).values(), lt('d'))
    far = tmp[(len(tmp) * int(the['Far']))]
    B = far['row']
    c = far['d']
    for n, two in enumerate(sort2(list(map(rows, proj).values()), lt('x'))):
        if n+1 <= (len(rows)+1) // 2:
            push(left, two['row'])
        else:
            push(right, two['row'])
    evals = 1 if the['Reuse'] and above else 2
    return left, right, A, B, c, evals


def tree(data, rows=None, cols=None, above=None):
    rows = rows or data.rows
    here = {'data': DATA.clone(data, rows)}
    if len(rows) >= 2*(len(data.rows)**the['min']):
        left, right, A, B, _,_ = half(data, rows, cols, above)
        here['left'] = tree(data, left, cols, A)
        here['right'] = tree(data, right, cols, B)
    return here


def showTree(tree, lvl=None):
    if tree:
        lvl = lvl if lvl is not None else 0
        res = '| ' * lvl + "["+str(len(tree['data'].rows))+"]" + '  '
        if 'left' not in tree or lvl == 0:
            print(res + str(o(stats(tree['data']))))
        else:
            print(res)
        if 'left' in tree:
            showTree(tree['left'], lvl+1)
        if 'right' in tree:
            showTree(tree['right'], lvl+1)


def sway(data):
    def worker(rows, worse, evals0, above=None):
        if len(rows) <= len(data.rows)**the['min']:
            return rows, many(worse, the['rest']*len(rows)),evals0
        else:
            l, r, A, B, _, evals = half(data, rows, None, above)
            if better(data, B, A):
                l, r, A, B = r, l, B, A

            def fun(row):
                push(worse, row)
            map(r, fun)
            return worker(l, worse,evals+evals0, A)
    best, rest, evals = worker(data.rows, {}, 0)
    return DATA.clone(data, best), DATA.clone(data, rest), evals



def bins(cols, rowss):
    def with1Col(col):
        n, ranges=withAllRows(col)
        ranges = sort2(map(ranges, itself).values(), lt('lo'))
        if type(ranges) == list:
            u = {}
            for k, v in enumerate(ranges):
                u[k] = v
            if col['isSym']: return u
            else: return merges(u,n/the['bins'], the['d']*div(col))
        else:
            if col['isSym']: return ranges
            else: return merges(ranges,n/the['bins'], the['d']*div(col))

    def withAllRows(col):
        n, ranges = 0, {}
        def xy(x , y , n):
            if x != '?':
                n += 1
                k = bin(col,x)
                ranges[k] = RANGE(col['at'], col['txt'], x) if k not in ranges else ranges[k]
                extend(ranges[k] , x , y)
        for y, rows in rowss.items():
            for _, row in rows.items():
                xy(row[col['at']],y,n)
        return n, ranges
    return map(cols, with1Col)



def oldbins(cols, rowss):
    out = {}
    for _, col in cols.items():
        ranges = {}
        for y, rows in rowss.items():
            for _, row in rows.items():
                x = row[col['at']]
                if x != '?':
                    k = int(bin(col, x))
                    ranges[k] = RANGE(col['at'], col['txt'],
                                      x) if k not in ranges else ranges[k]
                    extend(ranges[k], x, y)
        ranges = sort2(map(ranges, itself).values(), lt('lo'))
        if type(ranges) == list:
            u = {}
            for k, v in enumerate(ranges):
                u[k] = v
        out[len(out)] = u if col['isSym'] else mergeAny(ranges)
    return out


def bin(col, x):
    if x == '?' or col['isSym']:
        return x
    tmp = (col['hi'] - col['lo']) / (the['bins'] - 1)
    return 1 if col['hi'] == col['lo'] else math.floor(x/tmp + 0.5)*tmp


def merges(ranges0, nSmall, nFar):
    def noGaps(t: dict):
        for j in range(1, len(t)):
            t[j]['lo'] = t[j-1]['hi']
        t[0]['lo'] = float('-inf')
        t[len(t)-1]['hi'] = float('inf')
        if type(t) is list:
            u = {}
            for k, v in enumerate(t):
                u[k] = v
            return u
        return t
    def try2Merge(left,right,j):
        y = merged(left['y'], right['y'], nSmall, nFar)
        if y:
            j = j+1
            left['hi'], left['y']= right['hi'], y
        return j , left
    ranges1, j = {}, 0
    while j < len(ranges0):
        here = ranges0[j]
        if j < len(ranges0)-1:
            j,here = try2Merge(here, ranges0[j+1], j)
        j=j+1
        push(ranges1,here)
#   if len(ranges0) == len(ranges1):
#     print("here ranges0==ranges1: "+ str(type(noGaps(ranges0))))
#     return noGaps(ranges0)
#   else:
#     print("here ranges0!=ranges1: "+ str(type(mergeAny(ranges1))))
#     return mergeAny(ranges1)
    return noGaps(ranges0) if len(ranges0) == len(ranges1) else merges(ranges1, nSmall, nFar)


def merge2(col1, col2):
    new = merge(col1, col2)
    if div(new) <= (div(col1)*col1['n'] + div(col2)*col2['n'])/new['n']:
        return new

def merged(col1, col2, nSmall, nFar):
    new = merge(col1, col2)
    if nSmall and col1['n'] < nSmall or col2['n'] < nSmall:
        return new
    if nFar and not col1['isSym'] and math.abs(mid(col1)-mid(col2)) < nFar:
        return new
    if div(new) <= (div(col1)*col1['n'] + div(col2)*col2['n'])/new['n']:
        return new


def merge(col1, col2):
    new = copy(col1)
    if col1['isSym']:
        for x, n in col2['has'].items():
            add(new, x, n)
    else:
        for _, n in col2['has'].items():
            add(new, n)
        new['lo'] = min(col1['lo'], col2['lo'])
        new['hi'] = max(col1['hi'], col2['hi'])
    return new

def xpln(data,best,rest):
    def v(has):
        return value(has, len(best.rows), len(rest.rows), "best")
    def score(ranges):
        rule = RULE(ranges,maxSizes)
        if rule:
            oo(showRule(rule))
            bestr = selects(rule, best.rows)
            restr = selects(rule, rest.rows)
            if len(bestr) + len(restr) > 0:
                return v({'best':len(bestr), 'rest':len(restr)}) , rule
    tmp,maxSizes = {},{}
    for _,ranges in bins(data.cols['x'],{'best':best.rows, 'rest':rest.rows}).items():
        maxSizes[ranges[1]['txt']] = len(ranges)
        print('')
        for _,range in ranges.items():
            print(range['txt'], range['lo'], range['hi'])
            push(tmp, {'range':range, 'max':len(ranges),'val':v(range['y']['has'])})
    return firstN(sort3(tmp.values(),gt("val")),score)

def firstN(sortedRanges, scoreFun):
    srdict={}
    for i in range(len(sortedRanges)):
        srdict[i]=sortedRanges[i]
    print("")
    def tmpmpfun(r):
        print(str(r['range']['txt'])+"  "+str(r['range']['lo'])+"  "+str(r['range']['hi'])+"  "+str(rnd(r['val']))+"  "+o(r['range']['y']['has']))
    map(srdict,tmpmpfun)
    first = srdict[0]['val']
    def useful(range):
        if range['val'] > .05 and range['val'] > first/10:
            return range
    srdict = map(srdict, useful)
    srdictnew = {k: v for k, v in srdict.items() if v is not None}
    srdict.clear()
    srdict.update(srdictnew)
    out = {}
    most = -1
    for n in range(1,len(srdict)+1):
        if scoreFun(map(slice(srdict,0,n,1),on("range"))):
            tmp, rule = scoreFun(map(slice(srdict,0,n,1),on("range")))
        if tmp and tmp>most:
            out=rule
            most=tmp
    return out, most

def showRule(rule):

    def pretty(range):
        return range['lo'] if range['lo']==range['hi'] else {range['lo'], range['hi']}
    def merges(attr,ranges):
        sortedRanges = sort2(ranges.values(),lt("lo"))
        srdict={}
        for i in range(len(sortedRanges)):
            srdict[i]=sortedRanges[i]
        return map(merge(srdict),pretty),attr
    def merge(t0):
        t={}
        j=0
        while j<len(t0):
            left = t0[j]
            if j+1 not in t0.keys(): right = None
            else: right = t0[j+1]
            if right and left['hi'] == right['lo']:
                left['hi'] = right['hi']; j=j+1
            push(t, {'lo':left['lo'], 'hi':left['hi']})
            j=j+1
        return t if len(t0)==len(t) else merge(t)
    return kap(rule,merges)

def selects(rule, rows):
    def disjunction(ranges,row):
        for _,range in ranges.items():
            lo, hi, at = range['lo'], range['hi'], range['at']
            x = row[at]
            if x == "?":
                return True
            if lo == hi and lo == x:
                return True
            if lo <= x and x < hi:
                return True
        return False
    def conjunction(row):
        for _,ranges in rule.items():
            if not disjunction(ranges,row):
                return False
        return True
    def fun(r):
        if conjunction(r):
            return r
        else: return None
    mapped = map(rows, fun)
    d = {k: v for k, v in mapped.items() if v is not None}
    return d

def itself(x):
    return x


def rnd(n, nPlaces=2):
    mult = 10**(nPlaces or 2)
    return math.floor(n * mult + 0.5) / mult


Seed = 937162211


def rint(nlo=None, nhi=None):
    return math.floor(0.5 + rand(nlo, nhi))


def rand(nlo=None, nhi=None):
    global Seed
    nlo, nhi = nlo or 0, nhi or 1
    Seed = (16807 * Seed) % 2147483647
    return nlo + (nhi-nlo) * Seed / 2147483647


'''
-- Non-parametric effect-size test
--  M.Hess, J.Kromrey. 
--  Robust Confidence Intervals for Effect Sizes: 
--  A Comparative Study of Cohen's d and Cliff's Delta Under Non-normality and Heterogeneous Variances
--  American Educational Research Association, San Diego, April 12 - 16, 2004    
--  0.147=  small, 0.33 =  medium, 0.474 = large; med --> small at .2385
'''


def cliffsDelta(ns1, ns2):
    if len(ns1) > 256:
        ns1 = many(ns1, 256)
    if len(ns2) > 256:
        ns2 = many(ns2, 256)
    if len(ns1) > 10 * len(ns2):
        ns1 = many(ns1, 10 * len(ns2))
    if len(ns2) > 10 * len(ns1):
        ns2 = many(ns2, 10 * len(ns1))
    n, lt, gt = 0, 0, 0
    for _, x in ns1.items():
        for _, y in ns2.items():
            n += 1
            if x > y:
                gt += 1
            if x < y:
                lt += 1
    return abs(lt - gt) / n > the['cliffs']


# -- Given two tables with the same keys, report if their
# -- values are different.
def diffs(nums1, nums2):
    def diffsfun(k, nums):
        return cliffsDelta(nums['has'], nums2[k]['has']), nums['txt']
    return kap(nums1, diffsfun)

# String to thing

# -- Coerce string to boolean, int,float or (failing all else) strings.


def coerce(s):
    def fun(s1):
        if s1 == 'true':
            return True
        if s1 == 'false':
            return False
        return s1.strip()
    if s.isdigit():
        return int(s)
    try:
        tmp = float(s)
        return tmp
    except ValueError:
        return fun(s)


def Csv(fname, fun):
    n = 0
    with open(fname, 'r') as src:
        rdr = csv.reader(src, delimiter=',')
        for l in rdr:
            d = {}
            for v in l:
                d[len(d)] = coerce(v)
            n += len(d)
            fun(d)
    return n


# any; push `x` to end of list; return `x`


def push(t: dict, x):
    t[len(t)] = x
    return x


# t; return `t`,  sorted by `fun` (default= `<`)


def sort(t: dict, fun=lambda x: x.keys()):
    u = {}
    l = list(t.values())
    sl = sorted(l)
    for i in range(0, len(t)):
        u[i] = sl[i]
    return u


def sort2(t: list, fun=lambda x: x.keys()):
    return sorted(t, key=fun)

def sort3(t: list, fun=lambda x: x.keys()):
    return sorted(t, key=fun,reverse=True)

def on(x):
    def tmp(t):
        return t[x]
    return tmp

def lt(x: str):
    def fun(dic):
        return dic[x]
    return fun

def gt(x: str):
    def fun(dic):
        return dic[x]
    return fun
# x; returns one items at random


def any(t):
    return list(t.values())[rint(len(t), 1)-1]


# t1; returns some items from `t`
def many(t, n):
    u = {}
    for i in range(0, n):
        u[i] = any(t)
    return u


# t; map a function `fun`(v) over list (skip nil results)
def map(t: dict, fun):
    u = {}
    for k, v in t.items():
        u[k] = fun(v)
    return u


# ss; return list of table keys, sorted
def keys(t: list):
    return sorted(kap(t, lambda k, _: k))


# t; map function `fun`(k,v) over list (skip nil results)
def kap(t: dict, fun):
    u = {}
    for k, v in t.items():
        u[k] = fun(k, v)
    return u


# this function returns the pth element of the sorted list t
def per(t, p):
    if (p > 0 and p < 1):
        p = math.floor(len(t)*p)
    return t[p]


def copy(t):
    import copy
    return copy.deepcopy(t)  # not sure if it works


# Return a portion of `t`; go,stop,inc defaults to 1,#t,1.
# Negative indexes are supported.
def slice(t, go, stop, inc=1):
    if go and go < 0:
        go += len(t)
    if stop and stop < 0:
        stop += len(t)
    u = {}
    for j in range(int(go or 0), int(stop or len(t)), int(inc or 1)):
        u[len(u)]=t[j]
    return u


def fmt(sControl, *elements):  # emulate printf
    return (sControl % (elements))


def o(t, *isKeys):  # --> s; convert `t` to a string. sort named keys.
    if type(t) != dict:
        return str(t)

    def fun(k, v):
        if not re.findall('[^_]', str(k)):
            return fmt(":%s %s", o(k), o(v))

    if len(t) > 0 and not isKeys:
        tmp = map(t, o)
    else:
        tmp = sort(kap(t, fun))

    def concat(tmp: dict):
        res = []
        for k, v in tmp.items():
            res.append(':' + str(k))
            res.append(v)
        return res
    return '{' + ' '.join(concat(tmp)) + '}'


def oo(t):
    print(o(t))
    return t


def settings(s):  # --> t;  parse help string to extract a table of options
    t = {}
    # match the contents like: '-d  --dump  on crash, dump stack = false'
    res = r"[-][\S]+[\s]+[-][-]([\S]+)[^\n]+= ([\S]+)"
    m = re.findall(res, s)
    for key, value in m:
        t[key] = coerce(value)
    return t


def main(options, help, funs, *k):
    saved = {}
    fails = 0
    y, n = 0, 0
    for k, v in cli(settings(help), sys.argv).items():
        options[k] = v
        saved[k] = v
    if options['help']:
        print(help)

    else:
        for what, fun in funs.items():
            if options['go'] == 'all' or what == options['go']:
                for k, v in saved.items():
                    options[k] = v
                if fun() == False:
                    fails += 1
                    n += 1
                    print("❌ fail:", what)
                else:
                    y += 1
                    print("✅ pass:", what)
        if y+n > 0:
            print(
                ("\n🔆 " + str(o({'pass': y, 'fail': n, 'success': 100*y/(y+n)//1}))))


def cli(t, list):
    slots = list[1:]
    # search the key and value we want to update
    for slot, v in t.items():
        # give each imput slot an index(begin from 0)
        for n, x in enumerate(slots):
            # match imput slot with the.keys: x == '-e' or '--eg'
            if x == ('-'+slot[0]) or x == ('--'+slot):
                v = str(v)
                # we just flip the defeaults
                if v == 'True':
                    v = 'false'
                elif v == 'False':
                    v = 'true'
                else:
                    v = slots[n+1]
                t[slot] = coerce(v)
    return t


# Examples
egs = {}


def go(key, str, fun):  # --> nil; register an example.
    global help
    egs[key] = fun
    # help = help + f'  -g  {key}\t{str}\n'
    help = help + fmt('  -g  %s\t%s\n', key, str)


b5 = {}
if __name__ == '__main__':

    # eg("crash","show crashing behavior", function()
    #   return the.some.missing.nested.field end)
    def thefun():
        global the
        print(fmt("\n▶️  %s %s", 'the', ("-")*(60)))
        return oo(the)
    go("the", "show options", thefun)

    def randfun():
        print(fmt("\n▶️  %s %s", 'rand', ("-")*(60)))
        global Seed
        Seed = 1
        t = {}
        for _ in range(1, 1001):
            push(t, rint(100))
        Seed = 1
        u = {}
        for _ in range(1, 1001):
            push(u, rint(100))
        for k, v in t.items():
            assert (v == u[k])
    go("rand", "demo random number generation", randfun)

    def somefun():
        print(fmt("\n▶️  %s %s", 'some', ("-")*(60)))
        global the
        the['Max'] = 32
        num1 = NUM()
        for i in range(1, 10001):
            add(num1, i)
        oo(has(num1).values())
    go("some", "demo of reservoir sampling", somefun)

    def symsfun():
        print(fmt("\n▶️  %s %s", 'syms', ("-")*(60)))
        sym = adds(SYM(), ["a", "a", "a", "a", "b", "b", "c"])
        print(mid(sym), rnd(div(sym)))
        return 1.38 == rnd(div(sym))
    go("syms", "demo SYMS", symsfun)

    def numsfun():
        print(fmt("\n▶️  %s %s", 'nums', ("-")*(60)))
        num1, num2 = NUM(), NUM()
        for _ in range(1, 10001):
            add(num1, rand())
        for _ in range(1, 10001):
            add(num2, rand()**2)
        print("1  "+str(rnd(mid(num1), 1))+"   "+str(rnd(div(num1))))
        print("2  "+str(rnd(mid(num2)))+"   "+str(rnd(div(num2))))
        return .5 == rnd(mid(num1), 1) and mid(num1) > mid(num2)
    go("nums", "demo of NUM", numsfun)

    def csvfun():
        print(fmt("\n▶️  %s %s", 'csv', ("-")*(60)))
        n = 0

        def tmp(t):
            return len(t)
        n = Csv(the["file"], tmp)
        return n == 8*399
    go("csv", "reading csv files", csvfun)

    def datafun():
        print(fmt("\n▶️  %s %s", 'data', ("-")*(60)))
        data = DATA.read(the['file'])
        col = data.cols['x'][0]
        print(str(col['lo'])+" "+str(col['hi']) +
              " "+str(mid(col))+" "+str(div(col)))
        oo(stats(data))
    go("data", "showing data sets", datafun)

    def clonefun():
        print(fmt("\n▶️  %s %s", 'clone', ("-")*(60)))
        data1 = DATA.read(the['file'])
        data2 = data1.clone(data1.rows)
        oo(stats(data1))
        oo(stats(data2))
    go("clone", "replicate structure of a DATA", clonefun)

    def cliffsfun():
        print(fmt("\n▶️  %s %s", 'cliffs', ("-")*(60)))
        assert False == cliffsDelta({0: 8, 1: 7, 2: 6, 3: 2, 4: 5, 5: 8, 6: 7, 7: 3}, {
                                    0: 8, 1: 7, 2: 6, 3: 2, 4: 5, 5: 8, 6: 7, 7: 3}), "1"
        assert True == cliffsDelta({0: 8, 1: 7, 2: 6, 3: 2, 4: 5, 5: 8, 6: 7, 7: 3}, {
                                   0: 9, 1: 9, 2: 7, 3: 8, 4: 10, 5: 9, 6: 6}), "2"
        t1, t2 = {}, {}
        for _ in range(0, 1000):
            push(t1, rand())
        for _ in range(0, 1000):
            push(t2, math.sqrt(rand()))
        assert False == cliffsDelta(t1, t1), "3"
        assert True == cliffsDelta(t1, t2), "4"
        diff = False
        j = 1.0
        while (not diff):
            t3 = map(t1, lambda x: x*j)
            diff = cliffsDelta(t1, t3)
            print("> "+str(rnd(j))+"  "+str(diff))
            j = j*1.025
    go("cliffs", "stats tests", cliffsfun)

    def distfun():
        print(fmt("\n▶️  %s %s", 'dist', ("-")*(60)))
        data = DATA.read(the['file'])
        num = NUM()
        for _, row in data.rows.items():
            add(num, dist(data, row, data.rows[0], None))
        oo({'lo': num['lo'], 'hi': num['hi'],
           'mid': rnd(mid(num)), 'div': rnd(div(num))})
    go("dist", "distance test", distfun)

    def halffun():
        print(fmt("\n▶️  %s %s", 'half', ("-")*(60)))
        data = DATA.read(the['file'])
        left, right, A, B, c, _ = half(data)
        print(str(len(left))+"   "+str(len(right)))
        l = DATA.clone(data, left)
        r = DATA.clone(data, right)
        print("l   "+str(o(stats(l))))
        print("r   "+str(o(stats(r))))
    go("half", "divide data in half", halffun)

    def treefun():
        print(fmt("\n▶️  %s %s", 'tree', ("-")*(60)))
        showTree(tree(DATA.read(the['file'])))
    go("tree", "make snd show tree of clusters", treefun)

    def swayfun():
        print(fmt("\n▶️  %s %s", 'sway', ("-")*(60)))
        data = DATA.read(the['file'])
        best, rest, _ = sway(data)
        print("\nall    "+str(o(stats(data))))
        print("       "+str(o(stats(data, 'div'))))
        print("\nbest    "+str(o(stats(best))))
        print("       "+str(o(stats(best, 'div'))))
        print("\nrest    "+str(o(stats(rest))))
        print("       "+str(o(stats(rest, 'div'))))
        print("\nall != best?   " +
              str(o(diffs(best.cols['y'], data.cols['y']))))
        print("best != rest?   "+str(o(diffs(best.cols['y'], rest.cols['y']))))
    go("sway", "optimizing", swayfun)

    def binsfun():
        print(fmt("\n▶️  %s %s", 'bins', ("-")*(60)))
        b5 = ""
        data = DATA.read(the['file'])
        best, rest, _ = sway(data)
        print("all     " +
              str(o({'best': len(best.rows), 'rest': len(rest.rows)})))
        # print(bins(data.cols['x'],{'best':best.rows,'rest':rest.rows}))
        for k, t in bins(data.cols['x'], {'best': best.rows, 'rest': rest.rows}).items():
            for _, range in t.items():
                if range['txt'] != b5:
                    print("  ")
                b5 = range['txt']
                # print("this is range:")
                # print(range)
                print(str(range['txt'])+"  "+str(range['lo'])+"  "+str(range['hi'])+"  "+str(rnd(value(
                    range['y']['has'], len(best.rows), len(rest.rows), "best")))+"  "+str(o(range['y']['has'])))
    go("bins", "find deltas between best and rest", binsfun)


    def xplnfun():
        print(fmt("\n▶️  %s %s", 'xpln', ("-")*(60)))
        b5 = ""
        data = DATA.read(the['file'])
        best, rest, evals = sway(data)
        rule, most = xpln(data, best, rest)
        print("\n------------\nexplain="+"{:origin {3}}")
        data1 = DATA.clone(data,selects(rule,data.rows))
        print("all                 "+o(stats(data))+"  "+o(stats(data,div)))
        print(fmt("sway with %5s evals  ", evals)+o(stats(best))+"  "+o(stats(best,div)))
        print(fmt("xpln on   %5s evals  ", evals)+o(stats(data1))+"  "+o(stats(data1,div)))
        top, _ = betters(data, len(best.rows))
        top = DATA.clone(data, top)
        print(fmt("sort with %5s evals  ", len(data.rows))+o(stats(top))+"  "+o(stats(top,div)))
    go("xpln","explore explanation sets", xplnfun)
    main(the, help, egs)
