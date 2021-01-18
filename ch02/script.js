const temperature = [20, 21, 22, 23];
const sales = [40, 42, 44, 46];
const cause = tf.tensor(temperature);
const result = tf.tensor(sales);

const X = tf.input({ shape: [1] });
const Y = tf.layers.dense({ units: 1 }).apply(X);
const model = tf.model({ inputs: X, outputs: Y });
const compileParam = { optimizer: tf.train.adam(), loss: tf.losses.meanSquaredError };

model.compile(compileParam);

const fitParam = {
  epochs: 100,
  callbacks: {
    onEpochEnd: (epoch, logs) => {
      if (epoch / 10 === 0) {
        console.log(logs, 'RMSE: ', Math.sqrt(logs.loss));
      }
    }
  }
}, repeat = 101;

(async () => {
  for (let i = 0; i < repeat; i++) {
    console.log(`${i / repeat * 100}%`);
    await model.fit(cause, result, fitParam);
  }
})().then(() => {
  const predictedResult = model.predict(cause);
  predictedResult.print();
});
