let x_vals = []
let y_vals = []
let a, b, c;
let dragging = false;
const learningRate = 0.1;
const optimizer = tf.train.sgd(learningRate);

function setup() {
    createCanvas(500, 500)
    a = tf.variable(tf.scalar(random(-1, 1)))
    b = tf.variable(tf.scalar(random(-1, 1)))
    c = tf.variable(tf.scalar(random(-1, 1)))
}

function mousePressed() {
    dragging = true

}


function mouseReleased() {
    dragging = false;
}

function loss(pred, labels) {
    return pred.sub(labels).square().mean()
}

function predict(x) {
    const xs = tf.tensor1d(x)



    const ys = xs.square().mul(a).add(b.mul(xs)).add(c)
    return ys
}


function draw() {

    if (dragging) {
        let x = map(mouseX, 0, width, -1, 1)
        let y = map(mouseY, 0, height, 1, -1)
        x_vals.push(x)
        y_vals.push(y)
    } else {
        tf.tidy(() => {
            if (x_vals.length > 0) {
                const ys = tf.tensor1d(y_vals)
                optimizer.minimize(function() {
                    return loss(predict(x_vals), ys)
                })
            }

        })

    }

    background(0)


    stroke(255)
    strokeWeight(7)

    for (let i = 0; i < x_vals.length; i++) {
        let px = map(x_vals[i], -1, 1, 0, width)
        let py = map(y_vals[i], -1, 1, height, 0)
        point(px, py)
    }

    const curveX = []
    for (x = -1; x < 1.01; x += 0.05) {
        curveX.push(x)
    }
    const ys = tf.tidy(() => predict(curveX))
    let curveY = ys.dataSync()
    ys.dispose()

    beginShape()
    noFill()
    stroke(255)
    strokeWeight(2)
    for (i = 0; i < curveX.length; i++) {
        let x = map(curveX[i], -1, 1, 0, width)
        let y = map(curveY[i], -1, 1, height, 0)
        vertex(x, y)
    }
    endShape()


}