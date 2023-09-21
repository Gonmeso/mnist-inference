use core::traits::Into;
mod generated;
mod nn;

use array::{ArrayTrait, SpanTrait};
use debug::PrintTrait;
use nn::fc1;
use nn::fc2;
use generated::input::input;
use generated::fc1_bias::fc1_bias;
use generated::fc1_weights::fc1_weights;
use generated::fc2_bias::fc2_bias;
use generated::fc2_weights::fc2_weights;

use orion::operators::tensor::{TensorTrait, Tensor, I32Tensor};
use orion::numbers::i32;


fn main() -> Array<u32> {
    let input = input();
    let loops = *input.shape.at(0);
    let mut i: usize = 0;
    let mut predictions = ArrayTrait::<u32>::new();

    let fc1_bias = fc1_bias();
    let fc1_weights = fc1_weights();
    let fc2_bias = fc2_bias();
    let fc2_weights = fc2_weights();
    loop {
        if i > (loops - 1) {
            break();
        }
        let input_data_slice = input.data.slice(i * 196, 196);
        let t2 = TensorTrait::<i32>::new(array![196].span(), input_data_slice.into());
        let x = fc1(t2, fc1_weights, fc1_bias);
        let x = fc2(x, fc2_weights, fc2_bias);
        let x = *x.argmax(0, Option::None(()), Option::None(())).data.at(0);
        predictions.append(x);
        x.print();

        i += 1;
    };

    predictions
}