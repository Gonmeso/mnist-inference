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

fn main() -> Array<u32> {
    let input = input();
    let loops = *input.shape.at(1);
    let mut i: usize = 0;
    let mut predictions = ArrayTrait::<u32>::new();

    let fc1_bias = fc1_bias();
    let fc1_weights = fc1_weights();
    let fc2_bias = fc2_bias();
    let fc2_weights = fc2_weights();

    loop{
        if i > loops{
            break;
        }
        let input_slice = input.slice(
            starts: array![i, 0].span(),
            ends: array![i, 196].span(),
            axes: Option::Some(array![1].span()),
            steps: Option::None(())
            );
        input_slice.shape.len().print();
        input_slice.data.len().print();
        'Pre-flatten'.print();
        let t = input_slice.flatten(0);
        'Post-flatten'.print();
        let t2 = input_slice.reshape(array![196].span());
        let x = fc1(t2, fc1_weights, fc1_bias);
        let x = fc2(x, fc2_weights, fc2_bias);
        let x = *x.argmax(0, Option::None(()), Option::None(())).data.at(0);
        predictions.append(x);
    };


    predictions
}