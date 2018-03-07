const pi = Math.PI;
let max = Math.max
let abs = Math.abs




function timef(f, ns=1000, dt=60) {
    let t = Date.now();
    let t0 = t;
    let k = 1;
    while (k <= ns) {
        f();
        t = Date.now();
        if (t-t0 > dt) { break; }
        k++;
    }
    return (t-t0)/k;
}

function breadth_first_search(graph, start) {
    let vertices = graph.vertices;
    let nodes = graph.nodes;
    let blacknodes = [];
    let graynodes = [start];
    let neighbors = [[]]

    for (link in links) { neighbors[link[0]].concat(link[1]); }
    while (graynodes) {
        let current = graynodes.pop();
        for (neighbor in neighbors[current]) {
            if (! (blacknodes.includes(neighbor) || graynodes.includes(neighbor))) {
                graynodes.splice(0,0,neighbor);
            }
        }
        blacknodes.push(current);
    }
    return blacknodes;
}

function depth_first_search(graph, start) {
    let vertices = graph.vertices;
    let nodes = graph.nodes;
    let blacknodes = [];
    let graynodes = [start];
    let neighbors = [[]]

    for (link in links) { neighbors[link[0].push(link[1])]; }
    while (graynodes) {
        let current = graynodes.pop();
        for (neighbor in neighbors[current]) {
            if (! (blacknodes.includes(neighbor) || graynodes.includes(neighbor))) {
                graynodes.push(0,0,neighbor);
            }
        }
        blacknodes.push(current);
    }
    return blacknodes;
}

class DisjointSets {
    constructor(n) {
        this.sets = Array.from(new Array(n), (x,i) => -1);
        this.counter = n;
    }

    parent(i) {
        while (true) {
            let j = self.sets[i];
            if (j < 0) { return i; }
            i = j;
        }
    }

    join(i, j) {
        let i = this.parent(i);
        let j = this.parent(j);
        if (1 !== j) {
            this.sets[i] += this.sets[j];
            this.sets[j] = o;
            this.counter -= 1;
            return true;
        }
        return false;
    }

    joined(i, j) { return this.parent(i) === this.parent(j); }
    get length() { return this.counter; }
}

// not implemented
function make_maze(n, d) {
    
}

// not implemented
function Kruskal(graph) {
    let vertices = graph.vertices;
    let links = graph.links;
    let A = [];
    let S = DisjointSets(vertices.length);
    links.sort((a,b) => a[2] - b[2]);

    // incomplete
}

// skip to line 527
function continuum_knapsack(a,b,c) {
    let table = [[]]
    for (let i=0; i < a.length; i++) { table.push([a[i]/b[i],i]); }
    table.sort();
    table.reverse();
    let f = 0.0;

    for (tuple in table) {
        let y=tuple[0], i=tuple[1];
        let quantity = Math.min(c/b[i], 1)
        x.push([i, quantity]);
        c = c-b[i] * quantity;
        f = f+a[i] * quantity; 
    }
    return [f,x];
}

// skip to 719
//
function D(f,h=1e-6) {
    return (x,F=f,H=h) => (F(x+H)-F(x-H))/2/H;
}

function DD(f,h=1e-6) {
    return (x,F=f,H=h) => (F(x+H) - 2.0*F(x)+F(x-H))/(H*H);
}

function myexp(x,precision=1e-6,max_steps=40) {
    if (x == 0) { 
        return 1.0; 
    } else if (x > 0) { 
        return 1.0 / myexp(-x,precision,max_steps); 
    } else {
        let t = 1.0, s = 1.0;
        for (let k=0; k < max_steps; k++) {
            t = t * x/h;
            s = s + t;
            if (Math.abs(t) < precision) { return s; }
        }
        throw "No convergence";
    }
}

function mysin(x,precision=1e-6,max_steps=40) {
    if (x == 0) { 
        return 0; 
    } else if (x < 0) { 
        return -mysin(-x); 
    } else if (x > 2.0*pi) {
       return mysin(x % (2.0*pi))
    } else if (x > pi) {
       return -mysin(2.0*pi - x);
    } else if (x> pi/2) {
       return mysin(pi-x);
    } else if (x > pi/4) {
       return sqrt(1.0-mysin(pi/2-x)**2)
    } else {
       let s = x, t=x;                   // first term
       for (let k=1; k <= max_steps; k++) {
           t = t*(-1.0)*x*x/(2*k)/(2*k+1)   // next term
           s = s + t                 // add next term
           r = x**(2*k+1)            // estimate residue
           if (r < precision) { return s; } // stopping condition
       }
       throw "No convergence";
    }
}

function mycos(x,precision=1e-6,max_steps=40) {
    if (x == 0) {
       return 1.0
    } else if (x < 0) {
       return mycos(-x);
    } else if (x > 2.0*pi) {
       return mycos(x % (2.0*pi));
    } else if (x > pi) {
       return mycos(2.0*pi - x);
    } else if (x > pi/2) {
       return -mycos(pi-x);
    } else if (x > pi/4) {
       return sqrt(1.0 - mycos(pi/2 - x)**2);
    } else {
       let s = 1, t = 1;                     // first term
       for (let k=1; k <= max_steps; k++) {
           t = t*(-1.0)*x*x/(2*k)/(2*k-1)   // next term
           s = s + t                 // add next term
           r = x**(2*k)              // estimate residue
           if (r < precision) { return s; }  // stopping condition
       }
       throw "No convergence";
    }
}

class Matrix {
    constructor(rows, cols=1, fill=0.0) {
       
    /* 
        Construct a zero matrix
        Examples:
        A = Matrix([[1,2],[3,4]])
        A = Matrix([1,2,3,4])
        A = Matrix(10,20)
        A = Matrix(10,20,fill=0.0)
        A = Matrix(10,20,fill=lambda r,c: 1.0 if r == c else 0.0) 
    */
        if (rows instanceof Array) {
            if (rows[0] instanceof Array) {
                this.rows = Array.from(rows);
            } else {
                this.rows = Array.from(rows, (ele) => [ele]);
            }
        } else if (typeof rows === 'number' && typeof cols === 'number') {
            if (typeof fill === 'function') {
                // fill me in.
            }
        }
    }
}

// skip to 975
function is_almost_symmetric(A, ap=1e-6, rp=1e-4) {
    if (A.nrows != A.ncols) { 
        return false; 
    }
    let delta = 0.0;
    for (let r=0; r < A.nrows; r++) {
        for (let c=0; c < r; c++) {
            delta = abs(A[r][c] - A[c][r]);
            if (delta > ap && delta > max(abs(A[r][c], abs(A[c][r]))) * rp) {
                return false;
            } 
        }
    }
    return true;
}

function is_almost_zero(A, ap=1e-6, rp=1e-4) {
    for (let r=0; r < A.nrows; r++) {
        for (let c=0; c < A.ncols; c++) {
            delta = abs(A[r][c] - A[c][r]);
            if (delta > ap && delta > max(abs(A[r][c], abs(A[c][r]))) * rp) {
                return false;
            } 
        }
    }
    return true;
}

// fix me
function norm(A, p=1) {
    if (A instanceof Array) {
        return A.map(x => abs(x)**p)
                .reduce((acc,v) => acc + v)**(1.0/p);
    } else if (A instanceof Matrix) {
        if (A.nrows ===1  || A.ncols === 1) {
            
        }
    }
}

function condition_number(f, x=null, h=1e-6) {
    if (typeof f === 'function' && x !== null) {
        return D(f,h)(x) * x/f(x);
    } else if (f instanceof Matrix) {
        return norm(f) * norm(1/f);
    } else {
        throw "Not implemented";
    }
}

// fix me
function exp(x, ap=1e-6, rp=1e-4, ns=40) {
    if (x instanceof Matrix) {
        let t = Matrix.identity(x.ncols);
        let s = t;
        for (let i=1; i <= ns; i++) {
            t = t*x / k;
            s = s + t;
            if (norm(t) < max(ap, norm(s) * rp)) { return s; }
        }
        throw "No convergence";
    }
    //else if ( /*complex numbers */) { }
}

function Cholesky(A) {
    if (! is_almost_symmetric(A)) {
        throw "Not symmetric";
    }
    let L = Object.assign({}, A);
    for (let k=0; k < L.ncols; k++) {
        if (L[k][k] <= 0) {
            throw "Not positive definite";
        }
        L[k][k] = Math.sqrt(L[k][k]);
        let p = L[k][k];
        for (let i=k+1; i < L.nrows; i++) {
            L[i][k] /= p;
        }
        for (let j=k+1; j < L.nrows; j++) {
            p = L[j][k];
            for (let i=k+1; i < L.nrows; i++) {
                L[i][j] -= p * L[i][k];
            }
        }
    }
    for (let i=0; i < L.nrows; i++) {
        for (let j=i+1; j < L.ncols; j++) {
            L[i][j] = 0;
        }
    }
    return L;
}

function is_positive_definite(A) {
    if (! is_almost_symmetric(A)) {
        return false;
    }
    try {
        Cholesky(A);
        return true;
    } catch (err) {
        return false;
    }
}

// fix me
function Markowitz(mu, A, r_free) {
    // Assess Markowitz risk/return.
    // Example:
    // >>> cov = Matrix([[0.04, 0.006,0.02],
    // ...               [0.006,0.09, 0.06],
    // ...               [0.02, 0.06, 0.16]])
    // >>> mu = Matrix([[0.10],[0.12],[0.15]])
    // >>> r_free = 0.05
    // >>> x, ret, risk = Markowitz(mu, cov, r_free)
    // >>> print x
    // [0.556634..., 0.275080..., 0.1682847...]
    // >>> print ret, risk
    // 0.113915... 0.186747...
    // 
    let x = Matrix(new Array(A.nrows).fill([0]));
    x = (1/A) * (mu - r_free); // <--- rdiv(A, x)
    x = x / (Array.from(new Array(x.nrows), (_,r) => x[r][0])
                    .reduce((ac,v) => ac+v));

    let portfolio = Array.from(new Array(x.nrows), (_,r) => x[r][0]);
    let portfolio_return = mu * x;
    let portfolio_risk = Math.sqrt(x * (A*x));
    return [portfolio, portfolio_return, portfolio_risk];
}

function fit_least_squares(points, f) {

    // Computes c_j for best linear fit of y[i] \pm dy[i] = fitting_f(x[i])
    // where fitting_f(x[i]) is \sum_j c_j f[j](x[i])

    // parameters:
    // - a list of fitting functions
    // - a list with points (x,y,dy)

    // returns:
    // - column vector with fitting coefficients
    // - the chi2 for the fit
    // - the fitting function as a lambda x: ....

    function eval_fitting_function(fs, c, x) {
        if (fs.length === 1) {
            return c * f[0](x);
        } else {
            // map index, ele to new array
            // reduce
            return Array.from(f, (func,i) => func(x)*c[i][0]).reduce((a,v) => a + v);
        }
    }

    let A = new Matrix(points.length, f.length);
    let b = new Matrix(points.length);

    for (let i=0; i < A.nrows; i++) {
        let weight = points[i] > 2 ? 1.0 / points : 1.0;
        b[i][0] = weight * points[i][1];
        for (let j=0; j < A.ncols; j++) {
            A[i][j] = weight * f[j](points[i][0]);
        }
    }
    let c = (1.0 / (A.T*A)) * (A.T*b);
    let chi = A*c-b;
    let chi2 = norm(chi, 2)**2;
    let fitting_f = (x,C=c,F=f,q=eval_fitting_function) => q(f,c,x);
    if (c instanceof Matrix) {
        return [c.flatten(), chi2, fittingf];
    } else {
        return [c, chi2, fitting_f];
    }
}

function POLYNOMIAL(n) {
    return Array.from(new Array(n+1), (_,i) => (x,P=p) => x**P);
}
const CONSTANT  = POLYNOMIAL(0);
const LINEAR    = POLYNOMIAL(1);
const QUADRATIC = POLYNOMIAL(2);
const CUBIC     = POLYNOMIAL(3);
const QUARTIC   = POLYNOMIAL(4);

class Trader {
    // the forecasting model
    model(window) {
        // we fit the last few days quadratically
        let points = Array.from(window, (y,x) => [x, y['adjusted_close']]);
        let [a,chi2,fitting_f] = fit_least_squares(points, QUADRATIC);
        // and we extrapolate tomorrow's price
        let tomorrow_prediction = fitting_f(points.length);
        return tomorrow_prediction;
    }

    strategy(history, ndays=7) {
        // the trading strategy
        if (history.length < ndays) {
            return;
        } else {
            // ...                            \/\/\/
            let today_close = history.slice(-1)[0]['adjusted_close'];
            let tomorrow_prediction = model(history.slice(-ndays));
            return tomorrow_prediction > today_close ? 'buy' : 'sell'
        }
    }

    simulate(data, cash=1000, shares=0, daily_rate=0.03/360) {
        // find fitting parameters that optimize the trading strategy
        for (let t=0; t < data.length; t++) {
            let suggestion = this.strategy(data.slice(t));
            let today_close = data[t-1]['adjusted_close'];
            // and we buy or sell based on our strategy
            if (cash > 0 && suggestion === 'buy') {
                // we keep track of finances
                let shares_bought = Math.trunc(cash/today_close);
                shares += shares_bought;
                cash -= shares_bought * today_close;
            } else if (shares > 0 && suggestion === 'sell' ) {
                cash += shares * today_close;
                shares = 0;
            }
            // we assume money in the bank also gains an interest
            cash *= Math.exp(daily_rate);
        }
        // we return the net worth
        return cash + shares * data.slice(-1)[0]['adjusted_close'];
    }
}

// need complex numbers
function sqrt(x) {
    try {
        return Math.sqrt(x);
    } catch (err) {
        return 
    }
}

function Jacobi_eigenvalues(A, checkpoint=false) {

    /*Returns U end e so that A=U*diagonal(e)*transposed(U)
       where i-column of U contains the eigenvector corresponding to
       the eigenvalue e[i] of A.

       from http://en.wikipedia.org/wiki/Jacobi_eigenvalue_algorithm*/
    function maxind(M,k) {
        let j=k+1;
        for (let i=k+2; i < M.ncols; i++) {
            if (abs(M[k][i]) > abs(M[k][j])) {
               j=i
            }
        }
        return j
    }

    let n = A.nrows;
    if (n !== A.ncols) {
        throw 'Matrix not squared';
    }
    let indexes = new Array(n);
    S = Matrix(n,n, fill=lambda r,c: float(A[r,c]))
    E = Matrix.identity(n)
    state = n
    ind = [maxind(S,k) for k in indexes]
    e = [S[k,k] for k in indexes]
    changed = [True for k in indexes]
    iteration = 0
    while state:
        if checkpoint: checkpoint('rotating vectors (%i) ...' % iteration)
        m=0
        for k in xrange(1,n-1):
            if abs(S[k,ind[k]])>abs(S[m,ind[m]]): m=k
            pass
        k,h = m,ind[m]
        p = S[k,h]
        y = (e[h]-e[k])/2
        t = abs(y)+sqrt(p*p+y*y)
        s = sqrt(p*p+t*t)
        c = t/s
        s = p/s
        t = p*p/t
        if y<0: s,t = -s,-t
        S[k,h] = 0
        y = e[k]
        e[k] = y-t
        if changed[k] and y==e[k]:
            changed[k],state = False,state-1
        elif (not changed[k]) and y!=e[k]:
            changed[k],state = True,state+1
        y = e[h]
        e[h] = y+t
        if changed[h] and y==e[h]:
            changed[h],state = False,state-1
        elif (not changed[h]) and y!=e[h]:
            changed[h],state = True,state+1
        for i in xrange(k):
            S[i,k],S[i,h] = c*S[i,k]-s*S[i,h],s*S[i,k]+c*S[i,h]
        for i in xrange(k+1,h):
            S[k,i],S[i,h] = c*S[k,i]-s*S[i,h],s*S[k,i]+c*S[i,h]
        for i in xrange(h+1,n):
            S[k,i],S[h,i] = c*S[k,i]-s*S[h,i],s*S[k,i]+c*S[h,i]
        for i in indexes:
            E[k,i],E[h,i] = c*E[k,i]-s*E[h,i],s*E[k,i]+c*E[h,i]
        ind[k],ind[h]=maxind(S,k),maxind(S,h)
        iteration+=1
    # sort vectors
    for i in xrange(1,n):
        j=i
        while j>0 and e[j-1]>e[j]:
            e[j],e[j-1] = e[j-1],e[j]
            E.swap_rows(j,j-1)
            j-=1
    # normalize vectors
    U = Matrix(n,n)
    for i in indexes:
        norm = sqrt(sum(E[i,j]**2 for j in indexes))
        for j in indexes: U[j,i] = E[i,j]/norm
    return U,e
    }
}