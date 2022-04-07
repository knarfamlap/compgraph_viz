const assert = require("assert");

class Variable {
  constructor(value, local_grads = []) {
    this.value = value;
    this.local_grads = local_grads;
    this.grad = 0;
  }

  add(other) {
    other = typeof other === "number" ? new Variable(other, []) : other;
    assert(other instanceof Variable);

    let value = this.value + other.value;
    let local_grads = [
      [this, 1],
      [other, 1]
    ];

    return new Variable(value, local_grads);
  }

  sub(other) {
    other = typeof other === "number" ? new Variable(other, []) : other;
    assert(other instanceof Variable);
    return this.add(other.neg());
  }

  mul(other) {
    other = typeof other === "number" ? new Variable(other, []) : other;
    assert(other instanceof Variable);

    let value = this.value * other.value;
    let local_grads = [
      [this, other.value],
      [other, this.value]
    ];

    return new Variable(value, local_grads);
  }

  neg() {
    return this.mul(-1);
  }

  pow(val) {
    assert(typeof val === "number");
    let value = this.value ** val;
    let local_grads = [[this, val * this.value ** (val - 1)]];

    return new Variable(value, local_grads);
  }

  div(other) {
    other = typeof other === "number" ? new Variable(other, []) : other;
    assert(other instanceof Variable);

    return this.mul(other.pow(-1.0));
  }

  backward() {
    let grads = new Map();

    function compute_grads(node, path_val) {
      for (let i = 0; i < node.local_grads.length; i++) {
        var [child, local_gradient] = node.local_grads[i];

        let val_path_to_child = path_val * local_gradient;

        grads.set(child, grads.get(child) ?? 0 + val_path_to_child);
        child.grad = grads.get(child);
        compute_grads(child, val_path_to_child);
      }
    }
    compute_grads(this, 1);

    return grads;
  }
}

const a = new Variable(4.0);
const b = new Variable(2.0);
const f = a.div(b).pow(5.0);
console.log(f.backward());

console.log(b.grad);
