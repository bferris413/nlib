function timef(f, ns=1000, dt=60) {
    let t = Date.now();
    let t0 = t;
    let k=1;
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
    length() { return this.counter; }
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
function D(f, h=1e-6) {
    return (x,F=f,H=h) => (F(x+H)-F(x-H))/2/H;
}

function DD(f, h=1e-6) {
    return (x, F=f, H=h) => (F(x+H) - 2.0*F(x)+F(x-H))/(H*H);
}

function myexp(x, precision=1e-6, max_steps=40) {
    if (x==0) { 
        return 1.0; 
    } else if (x>0) { 
        return 1.0 / myexp(-x, precision, max_steps); 
    } else {
        let t = 1.0, s = 1.0;

    }
}