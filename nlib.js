/******************************************************************
 * nlib.js
 * 
 * Fork and translation of mdipierro's nlib.py numerical library
 * (https://github.com/mdipierro/nlib).
 *
 * CSC 331 - Undergraduate Project - Winter, 2018
 * By: Benjamin Ferris
 *****************************************************************/

const pi = Math.PI;
const max = Math.max
const abs = Math.abs

function div(a, b) {
    if (Number.isInteger(a) && Number,isInteger(b)) {
        return Math.trunc(a/b); // simulate integer division
    } else if (typeof a === 'number' && typeof b === 'number') {
        return a/b;
    } else {
        throw "Unsupported types";
    }
}

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
    let [vertices, links] = graph;
    let blacknodes = [];
    let graynodes = [start];
    let neighbors = Array(vertices.length).fill([]);
    for (let link of links) { 
        neighbors[link[0]].concat(link[1]); 
    }
    while (graynodes) {
        let current = graynodes.pop();
        for (let neighbor of neighbors[current]) {
            if (! (blacknodes.includes(neighbor) || graynodes.includes(neighbor))) {
                graynodes.splice(0,0,neighbor);
            }
        }
        blacknodes.push(current);
    }
    return blacknodes;
}

function depth_first_search(graph, start) {
    let [vertices, links] = graph;
    let blacknodes = [];
    let graynodes = [start];
    let neighbors = Array(vertices.length).fill([]);

    for (let link of links) {
        neighbors[link[0]].concat(link[1]); 
    }
    while (graynodes) {
        let current = graynodes.pop();
        for (let neighbor of neighbors[current]) {
            if (! (blacknodes.includes(neighbor) || graynodes.includes(neighbor))) {
                graynodes.push(neighbor);
            }
        }
        blacknodes.push(current);
    }
    return blacknodes;
}

class DisjointSets {
    constructor(n) {
        this.sets = new Array(n).fill(-1);
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
        [i, j] = [this.parent(i), this.parent(j)];
        if (i !== j) {
            this.sets[i] += this.sets[j];
            this.sets[j] = i;
            this.counter -= 1;
            return true; // they have been joined
        }
        return false; // they were already joined
    }

    joined(i, j) { 
        return this.parent(i) === this.parent(j); 
    }
    get length() {
        return this.counter; 
    }
}

// not implemented
function make_maze(n, d) {

}

function Kruskal(graph) {
    let [vertices, links] = graph;
    let A = [];
    let S = DisjointSets(vertices.length);
    links.sort((a,b) => a[2] - b[2]);
    for (let [source,dest,length] in links) {
        if (S.join(source, dest)) {
            A.push([source,dest,length]);
        }
    }
    return A;
}

class PrimVertex {
    constructor(id, links) {
        this.INFINITY = 1e100;
        this.id = id;
        this.closest = undefined;
        this.closest_dist = this.INFINITY;
        this.neighbors = links.filter(link => link[0] === id)
                                .map(link => link.splice(1));
    }

    // need javascript default comparison
    // lookup reference, this is wrong...
    cmp(me, you) {
        return this.closest_dist - you.closest_dist;
    }
}

// No heap implemented
function Prim(graph, start) {
    let [vertices, links] = graph;
    let P = vertices.map(i => new PrimVertex(i, links));
    let Q = vertices.filter(i => i !== start)
                    .map(i => P[i]);
    let vertex = P[start];
    while (Q) {
        let neighbor;
        for (let [neighbor_id, length] of vertex.neighbors) {
            neighbor = P[neighbor_id];
            if (Q.includes(neighbor) && length < neighbor.closest_dist) {
                neighbor.closest = vertex;
                neighbor.closest_dist = length;
            }
        }
        // need heap, fix
        heapify(Q);
        vertex = heappop(Q);
    }
    // need return, fix
}

function Dijkstra(graph, start) {

}

function encode_huffman(input) {

}

function decode_huffman(keys, encoded) {

}

function lcs(a, b) {
    let previous = Array(a.length).fill(0);
    let current, e;
    a.forEach((r,i) => {
        current = [];
        b.forEach((c,j) => {
            if (r === c) {
                e = i*j > 0 ? previous[j-1] + 1 : 1
            } else {
                e = Math.max(i > 0 ? previous[j] : 0,
                            j > 0 ? current[current.length - 1] : 0);
            }
            current.push(e);
        });
        previous = current;
    }) ;
    return current[current.length-1];
}

function needleman_wunsch(a, b, p=0.97) {
    let z = [];
    a.forEach((r,i) => {
        z.push([]);
        b.forEach((c,j) => {
            if (r === c) {
                e = i*j > 0 ? z[i-1][j-1] : 1;
            } else {
                e = p * Math.max(i > 0 ? z[i-1][j] : 0, 
                                j > 0 ? z[i][j-1] : 0);
            }
            z[z.length-1].push(e);
        });
    });
    return z;
}

function continuum_knapsack(a,b,c) {
    let table = Array(a.length).fill(0)
                                .map((_,i) => [Math.trunc(a[i]/b[i]), i]);
    table.sort(([l,m], [n,o]) => l-n);
    table.reverse();
    let f = 0;
    let quantity;
    table.forEach(([y,i]) => {
        quantity = Math.min(c/b[i], 1);
        x.push([i, quantity]);
        c = c - b[i]*quantity;
        f = f + a[i]*quantity;
    }) ;
    return [f,x];
}

class Cluster {
    constructor(points, metric, weights = undefined) {
        // need separate function for later call
        init(points, metric, weights);
    }

    init(points, metric, weights = undefined) {
        [this.points, this.metric] = [points, metric];
        this.k = points.length;
        this.w = weights ||Array(self.k).fill(1);

        // py implementation makes this a dict
        this.q = points.map((_,i) => [i]);
        this.d = [];
        let m;
        for (let i=0; i < this.k; i++) {
            for (let j=i+1; j < this.k; j++) {
                m = metric(points[i],points[j]);
                if (m !== undefined) {
                    this.d.push([m,i,j]);
                }
            }
        }
        this.d.sort((a,b) => a[0] - b[0]);
        this.dd = [];
    }

    parent(i) {
        let parent;
        while (Math.floor(i) === i) {
            [parent, i] = [i, this.q[i]];
        }
        return [parent, i];
    }

    step() {
        let i,j,x,y,new_d,old_d,a,b;
        if (this.k > 1) {
            // find new clusters to join
            [[this.r,i,j], this.d] = [this.d[0], this.d.map(e=>e).splice(1)]; // could use filter, see timing tests below
            // join them
            [i,x] = this.parent(i); // find members of cluster i
            [j,y] = this.parent(j); // find members of cluster j
            x += y;                 // join members
            this.q[j] = i;          // make j cluster point to i
            this.k -= 1;            // decrease cluster count
            // update all distances to new joined cluster
            new_d = []; // links not related to joined clusters
            old_d = new Map(); // old links related to joined cluster
            this.d.forEach(([r,h,k]) => {
                if ([i,j].includes(h)) {
                    [a,b] = old_d.has(k) ? old_d.get(k) : [0,0];
                    old_d.set(k, [a + this.w[k]*r, b + this.w[k]]);
                } else if ([i,j].contains(k)) {
                    [a,b] = old_d.has(h) ? old_d.get(h) : [0,0];
                    old_d.set(h, [a + this.w[h]*r, b + this.w[h]]);
                } else {
                    new_d.push([r,h,k]);
                }
            });
            new_d.concat([...old_d].map(([k, [a,b]]) => [a/b, i, k]));
            new_d.sort((a,b) => a[0] - b[0]);
            this.d = new_d;
            // update weight of new cluster
            this.w[i] = this.w[i] + this.w[j];
            // get new list of cluster members
            this.v = [...this.q.values()].filter(s => s instanceof Array);
            this.dd.push([this.r, self.v.length]);
        }
        return [this.r, this.v];
    }

    find(k) {
        // if necessary start again
        if (this.k < k) { 
            this.init(this.points, this.metric); 
        }
        while (this.k > k) {
            this.step(); 
        }
        return [this.r, this.v];
    }
}

class NeuralNetwork {
    /***********************************************************
    Back-Propagation Neural Networks
    Placed in the public domain.
    Original author: Neil Schemenauer <nas@arctrix.com>
    Modified by: Massimo Di Pierro
    Read more: http://www.ibm.com/developerworks/library/l-neural/
    ************************************************************/

    constructor(ni, nh, no) {
        // number of input, hidden, and output  nodes
        this.ni = ni + 1;
        this.nh = nh;
        this.no = no;

        // activates for nodes
        this.ai = Array(this.ni).fill(1);
        this.ah = Array(this.nh).fill(1);
        this.ao = Array(this.no).fill(1);

        // create weights
        this.wi = new Matrix(this.ni, this.nh, (r,c) => NeuralNetwork.rand(-.2,.2));
        this.wo = new Matrix(this.nh, this.no, (r,c) => NeuralNetwork.rand(-2,2));

        // last change in weights for momentum
        this.ci = new Matrix(this.ni, this.nh);
        this.co = new Matrix(this.nh, this.no);
    }

    update(inputs) {
        if (inputs.length !== this.ni-1) {
            throw "Wrong number of inputs";
        }

        // input activations
        for (let i=0; i < this.ni - 1; i++) {
            this.ai[i] = inputs[i];
        }

        // hidden activations
        let s;
        for (let j=0; j < this.nh; j++) {
            s = Array(this.ni).fill(undefined)
                                .map((_,i) => this.ai[i] * this.wi[i][j])
                                .reduce((a,v) => a+v);
            this.ah[j] = NeuralNetwork.sigmoid(s);
        }

        // output activations
        for (let k=0; k < this.no; k++) {
            s = Array(this.nh).fill(undefined)
                                .map((_,j) => this.ah[j] * this.wo[j][k])
                                .reduce((a,v) => a+v);
        }
        return [...this.ao];
    }

    back_propagate(targets, N, M) {
        if (targets.length !== this.no) {
            throw "Wrong number of target values";
        }

        // calculate error terms for output
        let output_deltas = Array(this.no).fill(0);
        let error;
        for (let  k=0; k < this.no; k++) {
            error = targets[k] = this.ao[k];
            output_deltas[k] = NeuralNetwork.dsigmoid(this.ao[k]) * error;
        }

        // calculate error terms for hidden
        let hidden_deltas = Array(this.nh).fill(0);
        for (let j=0; j < this.nh; j++) {
            error = Array(this.no).fill(undefined)
                                    .map((_,k) => output_deltas[k] * this.wo[j][k])
                                    .reduce((a,v) => a+v);
            hidden_deltas[j] = NeuralNetwork.dsigmoid(this.ah[j]) * error;
        }

        // update output weights
        let change;
        for (let j=0; j < this.nh; j++) {
            for (let k=0; k < this.no; k++) {
                change = output_deltas[k] * this.ah[j];
                this.wo[j][k] = this.wo[j][k] + N*change + M*this.co[j][k];
                this.co[j][k] = change;
            }
        }

        // update input weights
        for (let i=0; i < this.ni; i++) {
            for (let j=0; j < this.nh; j++) {
                change = hidden_deltas[j] * this.ai[i];
                this.wi[i][j] = this.wi[i][j] + N*change + M*this.ci[i][j];
                this.ci[i][j] = change;
            }
        }


        // calculate error
        error = Array(targets.length).fill(undefined)
                                    .map((_,k) => 0.5*(targets[k]-this.ao[k])**2)
                                    .reduce((a,v) => a+v);
        return error;
    }

    test(patterns) {
        for (let p of patterns) {
            console.log(`${p[0]} -> ${this.update(p[0])}`);
        }
    }

    weights() {
        console.log('Input weights:');
        for (let i=0; i < this.ni; i++) {
            console.log(this.wi[i]);
        }
        console.log('\nOutput weights:');
        for (let j=0; j < this.nh; j++) {
            console.log(this.wo[j]);
        }
    }

    train(patterns, iterations=1000, N=0.5, M=0.1, check=false) {
        // N: learning rate
        // M: momentum factor
        let error, inputs, targets;
        for (let i=0; i < iterations; i++) {
            error = 0;
            for (let p of patterns) {
                inputs = p[0];
                targets = p[1];
                this.update(inputs);
                error = error + this.back_propagate(targets, N, M);
            }
            if (check && i % 100 === 0) {
                console.log(`Error: ${error.toPrecision(14)}`);
            }
        }
    }

    static rand(a, b) {
        // Calculate a random number where:  a <= rand < b
        return (b-a) * Math.random() + a;
    }

    static sigmoid(x) {
        // Our sigmoid function, tanh is a littler nicer than the  standard 1/(1+e^-x)
        return Math.tanh(x);
    }

    static dsigmoid(y) {
        // Derivative of our sigmoid function, in terms of the output
        return 1 - y**2;
    }
}

function D(f,h=1e-6) {
    return (x,F=f,H=h) => (F(x+H)-F(x-H))/2/H;
}

function DD(f,h=1e-6) {
    return (x,F=f,H=h) => (F(x+H) - 2*F(x)+F(x-H))/(H*H);
}

function myexp(x,precision=1e-6,max_steps=40) {
    if (x == 0) { 
        return 1; 
    } else if (x > 0) { 
        return 1 / myexp(-x,precision,max_steps); 
    } else {
        let t = 1.0, s = 1.0; // first term
        for (let k=0; k < max_steps; k++) {
            t = t * x/k;    // next term
            s = s + t;      // add next term
            if (Math.abs(t) < precision) { return s; }
        }
        throw "No convergence";
    }
}

function mysin(x, precision=1e-6, max_steps=40) {
    if (x === 0) { 
        return 0; 
    } else if (x < 0) { 
        return -mysin(-x); 
    } else if (x > 2.0*pi) {
       return mysin(x % (2.0*pi));
    } else if (x > pi) {
       return -mysin(2.0*pi - x);
    } else if (x> pi/2) {
       return mysin(pi-x);
    } else if (x > pi/4) {
       return Math.sqrt(1.0-mysin(pi/2 - x)**2);
    } else {
       let s = x, t=x;                   // first term
       for (let k=1; k < max_steps; k++) {
           t = t*(-1.0)*x*x/(2*k)/(2*k+1)   // next term
           s = s + t                 // add next term
           r = x**(2*k+1)            // estimate residue
           if (r < precision) { return s; } // stopping condition
       }
       throw "No convergence";
    }
}

function mycos(x, precision=1e-6, max_steps=40) {
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
    constructor(rows, cols=1, filler=0) {
       
    /******************************************************* 
     * Construct a zero matrix
     * Examples:
     * A = Matrix([[1,2],[3,4]])
     * A = Matrix([1,2,3,4])
     * A = Matrix(10,20)
     * A = Matrix(10,20,fill=0.0)
     * A = Matrix(10,20,fill=lambda r,c: 1.0 if r == c else 0.0) 
    *************************************************************/
        if (rows instanceof Array) {
            if (rows[0] instanceof Array) {
                this.rows = rows.map(e => e);
            } else {
                this.rows = rows.map(e => [e]);
            }
        } else if (typeof rows === 'number' && typeof cols === 'number') {
            let [xrows, xcols] = [rows.length, cols.length];
            if (typeof filler === 'function') {
                this.rows = Array(xrows).fill(undefined)
                                        .map((_,r) => { 
                                            return Array(xcols).fill(undefined)
                                                                .map((_,c) => filler(r,c))
                                        });
                // both work, depends on reader. 
                // this.rows = Array(xrows).fill(Array(xcols).fill(undefined))
                //             .map((a,r) => {
                //                 return a.map((_,c) => filler(r,c))
                //             });
            } else {
                this.rows = Array(xrows).fill(Array(xcols).fill(filler));
            }
        } else {
            throw `Unable to build a matrix from ${rows}`
        }
        this.nrows = this.rows.length;
        this.ncols = this.rows[0].length;
    }
    static add(A, B) {
        let rows = []
        let col;
        if (A.nrows !== B.nrows || A.ncols != B.ncols) {
            throw "Incompatible dimensions";
        }
        for (let i=0; i < A.nrows; i++) {
            col = []
            for (let j=0; j < A.ncols; j++) {
                col.push(A[i][j] + B[i][j]);
            }
            rows.push(col);
        }
        return new Matrix(rows);
    }
}

function is_almost_symmetric(A, ap=1e-6, rp=1e-4) {
    if (A.nrows != A.ncols) { 
        return false; 
    }
    let delta = 0.0;
    for (let r=0; r < A.nrows; r++) {
        for (let c=0; c < r; c++) {
            delta = abs(A.rows[r][c] - A.rows[c][r]);
            if (delta > ap && delta > max(abs(A.rows[r][c], abs(A.rows[c][r]))) * rp) {
                return false;
            } 
        }
    }
    return true;
}

function is_almost_zero(A, ap=1e-6, rp=1e-4) {
    for (let r=0; r < A.nrows; r++) {
        for (let c=0; c < A.ncols; c++) {
            delta = abs(A.rows[r][c] - A.rows[c][r]);
            if (delta > ap && delta > max(abs(A.rows[r][c], abs(A.rows[c][r]))) * rp) {
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

// Need a complex number library
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
        if (L.rows[k][k] <= 0) {
            throw "Not positive definite";
        }
        L.rows[k][k] = Math.sqrt(L.rows[k][k]);
        let p = L.rows[k][k];
        for (let i=k+1; i < L.nrows; i++) {
            L.rows[i][k] /= p;
        }
        for (let j=k+1; j < L.nrows; j++) {
            p = L.rows[j][k];
            for (let i=k+1; i < L.nrows; i++) {
                L.rows[i][j] -= p * L.rows[i][k];
            }
        }
    }
    for (let i=0; i < L.nrows; i++) {
        for (let j=i+1; j < L.ncols; j++) {
            L.rows[i][j] = 0;
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
    x = (1/A) * (mu - r_free); // <--- rdiv(A, x), need support
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
        let weight = points[i] > 2 ? 1 / points : 1;
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

// fix
function compute_correlation(stocks, key = 'arithmetic_return') {
    // The input must be a list of YStock(...).historical() data
    // find trading days common to all stocks
    let days = new Set();
    let nstocks = stocks.length;
}

function invert_minimum_residual(f,x,ap=1e-4,rp=1e-4,ns=200) {
    let y = x;
    let r = x - 1 * f(x);
    for (let k=0; k < ns; k++) {
        let q = f(r);
        let alpha = (q*r) / (q*q);
        y = y + alpha*r;
        r = r - alpha*q;
        let residue = Math.sqrt((r*r) / r.nrows);
        if (residue < Math.max(ap, norm(y) * rp)) {
            return y;
        }
    }
    throw "No convergence";
}

function invert_bicgstab(f, x, ap = 1e-4, rp = 1e-4, ns = 200) {
    let y = x;
    let r = x - 1.0 * f(x);
    let q = r;
    let p = 0;
    let s = 0;
    let rho_old = alpha = omega = 1;
    for (let k = 0; k < ns; k++) {
        let rho = q*r;
        let beta = (rho/rho_old) * (alpha/omega);
        rho_old = rho;
        p = beta*p + r - (beta*omega)*s;
        s = f(p);
        alpha = rho/(q*s);
        r = r - alpha * s;
        t = f(r);
        omega = (t*r) / (t*t);
        y = y + omega * r + alpha * p;
        let residue = Math.sqrt((r*r) / r.nrows);
        if (residue < max(ap, norm(y) * rp)) {
            return y;
        }
        r = r - omega * t;
    }
    throw "No convergence";
}

// check to repeat declarations
function solve_fixed_point(f, x, ap = 1e-6, rp = 1e-4, ns = 100) {
    let g = x => f(x) + x;
    let Dg = D(g);
    for (let k=0; k < ns; k++) {
        if (Math.abs(Dg(x)) >= 1) { throw "error D(g)(x) >= 1"; }
        let [x_old, x] = [x, g(x)];
        if (k > 2 && norm(x_old - x) < Math.max(ap, norm(x) * rp)) {
            return x;
        }
    }
    throw "No convergence";
}

function solve_bisection(f, a, b, ap = 1e-6, rp = 1e-4, ns = 100) {
    let [fa, fb] = [f(a), f(b)];
    if (fa === 0) return a;
    if (fb === 0) return b;
    if (fa * fb > 0) throw "f(a) and f(b) must have opposite sign";

    for (let k=0; k < ns; k++) {
        let x = (a+b) / 2;
        let fx = f(x);
        if (fx === 0 || norm(b-a) < Math.max(ap, norm(x) * rp)) return x;
        else if (fx * fa < 0) [b,fb] = [x, fx];
        else [a, fa] = [x, fx];
    }
    throw "No convergence";
}

function solve_newton(f, x, ap = 1e-6, rp = 1e-4, ns = 20) {
    for (let k = 0; k < ns; k++) {
        let [fx, Dfx] = [f(x), D(f)(x)];

        if (norm(Dfx) < ap) { throw "Unstable solution"; }

        let [x_old, x] = [x, x - (fx/Dfx)];
        if (k > 2 && norm(x - x_old) < Math.max(ap, norm(x) * rp)) { return x; }
    }
    throw "No convergence";
}

function solve_secant(f, x, ap = 1e-6, rp = 1e-4, ns = 20) {
    let [fx, Dfx] = [f(x), D(f)(x)];
    for (let k=0; k < ns; k++) {
        if (norm(Dfx) < ap) { 
            throw "Unstable solution"; 
        }
        let [x_old, fx_old, x] = [x, fx, x - (fx/Dfx)];
        if (k > 2 && norm(x - x_old) < Math.max(ap, norm(x) * rp)) {
            return x;
        }
        fx = f(x);
        Dfx = (fx - fx_old) / (x - x_old);
    }
    throw "No convergence";
}

function optimize_bisection(f, a, b, ap = 1e-6, rp = 1e-4, ns = 100) {
    return solve_bisection(D(f), a, b, ap, rp, ns);
}

function optimize_newton(f, x, ap = 1e-6, rp = 1e-4, ns = 100) {
    let Df = DD(f);
    f = D(f);
    for (let k=0; k < ns; k++) {
        let [fx, Dfx] = [f(x), Df(x)];
        if (Dfx === 0) { return x; }
        if (norm(Dfx) < ap) { throw "Unstable solution"; }
        let x_old = x ;
        x = x - fx/Dfx;
        if (norm(x - x_old) < Math.max(apl, norm(x) * rp)) {
            return x;
        }
    }
    throw "No convergence";
}

function optimize_secant(f, x, ap = 1e-6, rp = 1e-4, ns = 100) {
    let Df = DD(f);
    f = D(f);
    let [fx, Dfx] = [f(x), Df(x)];
    for (let k=0; k < ns; k++) {
        if (fx === 0) return x;
        if (norm(Dfx) < ap) throw "Unstable solution";
        let [x_old, fx_old] = [x, fx];
        x = x - fx/Dfx;
        if (norm(x - x_old) < Math.max(ap, norm(x) * rp)) {
            return x;
        }
        fx = f(x);
        Dfx = (fx - fx_old) / (x - x_old);
    }
    throw "No convergence";
}

function optimize_golden_search(f, a, b, ap = 1e-6, rp = 1e-4, ns = 100) {
    let tau = (Math.sqrt(5) - 1) / 2
    let [x, x2] = [a + (1-tau)*(b-a), a + tau*(b-a)];
    let [fa, f1, f2, fb] = [f(a), f(x1), f(x2), f(b)];
    for (let k=0; k < ns; k++) {
        if (f1 > f2) {
            [1, fa, x1, f1] = [x1, f1, x2, f2];
            x2 = a + tau*(b-a);
            f2 = f(x2);
        } else {
            [b, fb, x2, f2] = [x2, f2, x1, f1];
            x1 = a + (1-tau)*(b-a);
            f1 = f(x1);
        }
        if (k > 2 && norm(b-a) < Math.max(ap, norm(b) * rp)) {
            return b;
        }
    }
    throw "No convergence";
}

function partial(f, i, h = 1e-4) {
    let [F,I,H] = [f,i,h];
    let df = (x, f=F, i=I, h=H) => {
    // skip, need list(item);
    }
}

function gradient(f, x, h = 1e-4) {
    return new Matrix(x.length, x.length, (r,c) => partial(f,r,h)(x));
}

function hessian(f, x, h=1e-4) {
    return new Matrix(x.length, x.length, (r,c) => partial(partial(f,r,h),c,h)(x));
}

function jacobian(f, x, h = 1e-4) {
    let partials = Array.from(new Array(x.length), (_,c) => partial(f,c,h)(x));
    return new Matrix(partials[0].length, x.length, (r,c) => partials[c][r]);
}

function solve_newton_multi(f, x, ap = 1e-6, rp = 1e-4, ns = 20) {
    /***************************************************************
    Computes the root of a multidimensional function f near point x.

    Parameters
    f is a function that takes a list and returns a scalar
    x is a list

    Returns x, solution of f(x)=0, as a list
    ***************************************************************/
   let n = x.length;
   x = new Matrix(x.length);
   for (let k =0; k < ns; k++) {
       let fx = new Matrix(f(x.flatten()));
       let J = jacobian(f, x.flatten());
       if (norm (J) < ap) {
           throw "Unstable solution";
       }
       let x_old = x;
       x = x-(1 / J) * fx;
       if (k > 2 && norm(x - x_old) < Math.max(ap, norm(x) * rp)) {
           return x.flatten();
       }
   }
   throw "No convergence";
}

function optimize_newton_multi(f, x, ap = 1e-6, rp = 1e-4, ns = 20) {

    /************************************************************
    Finds the extreme of multidimensional function f near point x.

    Parameters
    f is a function that takes a list and returns a scalar
    x is a list

    Returns x, which maximizes of minimizes f(x)=0, as a list
    ************************************************************/
    x = new Matrix(Array.from(x));
    for (let k=0; k < n; k++) {
        let [grad, H] = [gradient(f, x.flatten()), hessian(f, x.flatten())];
        if (norm(H) < ap) {
            throw "Unstable solution";
        }
        let x_old = x;
        x = x - (1 / H) * grad;
        if (k > 2 && norm(x - x_old) < Math.max(ap, norm(x) * rp)) {
            return x.flatten();
        }
    }
    throw "No convergence";
}

function optimize_newton_multi_improved(f, x, ap = 1e-6, rp = 1e-4, ns = 20, h = 10) {

    /************************************************************
    Finds the extreme of multidimensional function f near point x.

    Parameters
    f is a function that takes a list and returns a scalar
    x is a list

    Returns x, which maximizes of minimizes f(x)=0, as a list
    ************************************************************/
    x = new Matrix(Array.from(x));
    let fx = f(x.flatten());
    for (let k=0; k < n; k++) {
        let [grad, H] = [gradient(f, x.flatten()), hessian(f, x.flatten())];
        if (norm(H) < ap) {
            throw "Unstable solution";
        }
        let [fx_old, x_old] = [fx, x];
        x = x - (1/H)*grad;
        fx = f(x.flatten());
        while (fx > fx_old) { // revert to steepest descent
            [fx, x] = [fx_old, x_old];
            let norm_grad = norm(grad);
            [x_old, x] = [x, x - grad/norm_grade*h];
            [fx_old, fx] = [fx, f(x.flatten())];
            h = h/2;
        }
        h = norm(x - x_old)*2;
        if (k > 2 && h/2 < Math.max(ap, norm(x) * rp)) {
            return x.flatten();
        }
    }
    throw "No convergence";
}

// fix
function fit(data, fs, b=undefined, ap = 1e-6, rp = 1e-4, ns = 200, constraint = undefined) {
    if (! fs instanceof Array) {
        let Data = data;
        let Constraint = constraint;
        function g(b, data = Data, f = fs, contraint = Constraint) {
            let chi2 = Array.from(data, ([x,y,dy]) => ((y-f(b,x))/dy)**2)
                            .reduce((acc,v) => acc + v);
            if (constraint) {
                chi2 += constraint(b);
            }
            return chi2;
        }
        if (b instanceof Array) {
            b = optimize_newton_multi_improved(g,b,ap,rp,ns);
        } else {
            b = optimize_newton(g,b,ap,rp,ns);
        }
        // semantics?
        return [b, g(b,data,undefined,undefined)];
    } else if (! b) {
        let [a, chi2, ff] = fit_least_squares(data, fs);
    } else {
        let na = fs.length;
        let Data = data;
        let Fs = fs;
        function core(b, data = Data, fs = Fs) {
            let A = new Matrix()
            // unfinished
        }
    }
}

function integrate_naive(f, a, b, n=20) {
    /***************************************************
    Integrates function, f, from a to b using the trapezoidal rule
    >>> from math import sin
    >>> integrate(sin, 0, 2)
    1.416118...
    ***************************************************/
    const h = Math.trunc((b-a) / n);
    return h/2 * (f(a) + f(b)) + h * (Array.from(new Array(n-1), (_,i) => f(a+h*i))
                                        .reduce((ac,v) => ac + v));
}

function integrate(f, a, b, ap = 1e-4, rp = 1e-4, ns = 20) {
    /****************************************************
    Integrates function, f, from a to b using the trapezoidal rule
    converges to precision
    ****************************************************/
    let I = integrate_naive(f, a, b, 1), I_old;
    for (let k=1; k < ns; k++) {
        [I_old, I] = [I, integrate_naive(f, a, b, 2**k)];
        if (k > 2 && norm(I-I_old) < Math.max(ap, norm(I) * rp)) {
            return I;
        }
    }
    throw "No convergence";
}

// Matrix divs, fix
class QuadratureIntegrator {
    /**********************************************************
    Calculates the integral of the function f from points a to b
    using n Vandermonde weights and numerical quadrature.
    ***********************************************************/
   constructor(order = 4) {
       let h = 1/(order=1);
       let A = new Matrix(order, order, (r,c) => (c*h)**r);
       let s = new Matrix(order, 1, (r,c) => 1 / (r+1));
       // Matrix div, fix
       let w = (1/A) * s;
       this.w = w;
   }

   integrate(f, a, b) {
       let w = self.w;
        let order = w.rows.length;
        let h = (b-a)/(order-1);
        return (b-a) * Array.from(new Array(order), (_,i) => w[i][0] * f(a + i*h))
                            .reduce((ac,v) => av + v);
   }
}

function integrate_quadrature_naive(f, a, b, n = 20, order = 4) {
    let h = (b-a) / n;
    let q = QuadratureIntegrator(order);
    return Array.from(new Array(n), (_,i) => q.integrate(f, a+i*h, a+i*h+h))
                .reduce((ac,v) => ac + v);
}

let E = (f,S) => (S.map(x => f(x)).reduce((ac,v) => ac + v)) / (S.length || 1);
let mean = X => E(x => x, X);
let variance = X => E(x => x**2, X) - E(x => x, X)**2;
let sd = X => Math.sqrt(variance(X));

function covariance(X, Y) {
    let sum = new Array(X.length)
                .fill(0)
                .map((_,i) => X[i] * Y[i])
                .reduce((a,v) => a + v);
    return sum/X.length - mean(X) * mean(Y);
}

let correlation = (X,Y) => covariance(X,Y) / sd(X) / sd(Y);

class MCG {
    constructor(seed, a=66539, m=2**31) {
        this.x = seed;
        [this.a, this.m] = [a, m];
    }

    next() {
        this.x = (this.a * this.x) % this.m;
        return this.x;
    }

    random() { 
        return this.next() / this.m;
    }
}

// skip Mersenne Twister - down to 1619
function leapfrog(mcg, k) {
    let a = mcg.a**k % mcg.m;
    return new Array(k).fill(0).map(_ => new MCG(mcg.next(), a, mgc.m));
}

class RandomSource {
    constructor(generator = undefined) {
        if (! generator) { this.generator = Math.random; }
        else { this.generator = generator; }
    }

    random() { 
        return this.generator(); 
    }

    randint(a,b) { 
        return Math.trunc(a + (b-a+1) * this.generator()); 
    }

    choice(S) {
        return S[this.randint(0, S.length - 1)];
    }

    bernoulli(p) {
        return this.generator() < p ? 1 : 0;
    }

    // fix
    lookup(table, epsilon = 1e-6) {
        let u = this.generator();
        // you can take an obj
        for (let key in table) {

        }
    }

    binomial(n, p, epsilon = 1e-6) {
        let u = this.generator();
        let q = (1-0)**n;
        for (let k=0; k < n+1; k++) {
            if (u < q+epsilon) {
                return k;
            }
            u = u-q;
            q = q * (n-k) / (k+1) * p / (1-p);
        }
        throw "Invalid probability";
    }

    negative_binomial(k, p, epsilon = 1e-6) {
        let u = this.generator();
        let n= k;
        let q = p**k;
        while (true) {
            if (u < q+epsilon) {
                return n;
            }
            u = u-q;
            q = q * n / (n-k+1) * (1-p);
            n = n+1;
        }
        throw "Invalid probability";
    }

    poisson(lamb, epsilon = 1e-6) {
        let u = this.generator();
        let q = exp(-lamb);
        let k = 0;
        while (true) {
            if (u < q+epsilon) {
                return k;
            }
            u = u-q;
            q = q*lamb/(k+1);
            k = k+1;
        }
        throw "Invalid probability";
    }

    uniform(a, b) {
        return a+(b-a) * this.generator();
    }

    exponential(lamb) {
        return -Math.log(this.generator()) / lamb;
    }

    // fix
    gauss(mu = 0, sigma = 1) {

    }

    pareto(alpha, xm) {
        let u = this.generator();
        return xm * (1-u)**(-1/alpha);
    }

    point_on_circle(radius = 1) {
        let angle = 2 * pi * this.generator();
        return [radius * Math.cos(angle), radius * Math.sin(angle)]
    }

    point_in_circle(radius = 1) {
        while (true) {
            let x = this.uniform(-radius, radius);
            let y = this.uniform(-radius, radius);
            if (x*x + y*y < radius*radius) {
                return [x,y,z];
            }
        }
    }

    point_in_sphere(radius = 1) {
        while (true) {
            let x = this.uniform(-radius, radius);        
            let y = this.uniform(-radius, radius);        
            let z = this.uniform(-radius, radius);        

            if (x*x + y*y + z*z < radius*radius) {
                return [x,y,z];
            }
        }
    }

    point_on_sphere(radius = 1) {
        let [x,y,z] = this.point_in_sphere(radius);
        let norm = Math.sqrt(x*x + y*y + z* z);
        return [x/norm, y/norm, z/norm];    
    }
}

function confidence_intervals(mu, sigma) {
    // Computes the  normal confidence intervals.
    const CONFIDENCE = [
        [0.68,1.0],
        [0.80,1.281551565545],
        [0.90,1.644853626951],
        [0.95,1.959963984540],
        [0.98,2.326347874041],
        [0.99,2.575829303549],
        [0.995,2.807033768344],
        [0.998,3.090232306168],
        [0.999,3.290526731492],
        [0.9999,3.890591886413],
        [0.99999,4.417173413469]
    ];
    return Array.from(CONFIDENCE, ([a,b]) => [a, mu-b*sigma, mu+b*sigma]);
}

function resample(S, size=undefined) {
    return Array(size || S.length)
                .fill(0)
                .map(_ => S[Math.floor(Math.random() * S.length)]);
}

function bootstrap(x, confidence = .68, nsamples = 100) {
    // Computes the bootstrap errors of the input list.
    let mean = S => S.reduce((a,v) => a + v) / S.length;
    let means = Array(nsamples)
                    .fill(0)
                    .map(_ => mean(resample(x)));
    means.sort((a,b) => a-b);
    let left_tail = Math.trunc(((1.0-confidence) / 2) * nsamples);
    let right_tail = nsamples - 1 - left_tail;
    return [means[left_tail], mean(x), means[right_tail]];
}

class MCEngine {
    constructor() {
        // empty on purpose
    }

    simulate_many(ap = .1, rp = .1, ns = 1000) {
        this.results = [];
        let s1 = 0, s2 = 0;
        this.convergence = false;
        let x,mu,variance,dmu;
        for (let k=1; k < ns; k++) {
            x = this.simulate_once();
            this.results.append(x);
            s1 += x;
            s2 += x*x;
            mu = s1/k;
            variance = s2/k - mu*mu;
            dmu = Math.sqrt(variance/k);
            if (k > 10) {
                if (Math.abs(dmu) < Math.max(ap, Math.abs(mu) * rp)) {
                    // bug here, Python code reads 'converence'
                    this.convergence = true;
                    break;
                }
            }
        }
        this.results.sort((a,b) => a-b);
        return bootstrap(this.results);
    }

    varr(confidence = 95) {
        let index = Math.trunc(0.01 * this.results.length * confidence + 0.999);
        if (this.results.length - index < 5) {
            throw "Not enough data, not reliable";
        }
        return this.results[index];
    }
}


/*****************************************************
 * Timing comparison of functions creating new arrays.
 * ***************************************************/
function timing() {

    let A = []
    for (let i=0; i < 10000; i++) {
        A.push(new Array(1000).fill(1000));
    }
    B = []

    // by far the slowest
    console.log('Array.from(...):')
    console.time('from');
    for (let i=0; i < A.length; i++) {
        B.push(Array.from(A).slice(1));
    }
    console.timeEnd('from');

    while (B.length != 0) {
        B.pop();
    }
    
    console.log('[...array]:')
    console.time('...array');
    for (let i=0; i < A.length; i++) {
        B.push([...A].slice(1));
    }
    console.timeEnd('...array');

    while (B.length != 0) {
        B.pop();
    }

    // fastest, should use when space isn't an issue (300ms faster than filter,
    // at the cost of creating an additional array).
    console.log('...map(_ => ...):')
    console.time('map');
    for (let i=0; i < A.length; i++) {
        B.push(A.map(e => e).slice(1));
    }
    console.timeEnd('map');

    while (B.length != 0) {
        B.pop();
    }

    // second fastest, should be used if space is an issue (creates one less
    // array than map, but (I think) incurs a slowdown on conditional check).
    console.log('.filter(_ => ...):')
    console.time('filter');
    for (let i=0; i < A.length; i++) {
        B.push(A.filter((_,i) => i > 0));
    }
    console.timeEnd('filter');

    // // same speed as below
    // console.log('Array(i).fill(i).map( =>...)');
    // console.time('fill');
    // for (let i=1; i <= 20000; i++) {
    //     Array(i).fill(undefined).map((_,i) => i**2).reduce((a,v) => a+v);
    // }
    // console.timeEnd('fill');

    // // fastest
    // console.log('new Array(i).fill(i)...');
    // console.time('fillnew');
    // for (let i=1; i <= 20000; i++) {
    //     new Array(i).fill(undefined).map((_,i) => i**2).reduce((a,v) => a+v);
    // }
    // console.timeEnd('fillnew');

    // // best syntax, average run-time of all options
    // console.log('[...Array(i)].map( =>...)');
    // console.time('spread');
    // for (let i=1; i <= 20000; i++) {
    //     [...Array(i)].map((_,i) => i**2).reduce((a,v) => a+v);
    // }
    // console.timeEnd('spread');

    // // longest syntax, and by far the slowest run time
    // console.log('Array.from(Array(i)...).map( =>...)');
    // console.time('from');
    // for (let i=1; i <= 20000; i++) {
    //     Array.from(Array(i), e => undefined).map((_,i) => i**2).reduce((a,v) => a+v);
    // }
    // console.timeEnd('from');
}

