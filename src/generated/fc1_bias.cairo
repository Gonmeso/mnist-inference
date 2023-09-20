use array::ArrayTrait;
use orion::operators::tensor::{TensorTrait, Tensor, I32Tensor};
use orion::numbers::i32;


fn fc1_bias() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(10);
    let mut data = ArrayTrait::<i32>::new();
    data.append(i32 { mag: 2912, sign: true });
    data.append(i32 { mag: 9303, sign: false });
    data.append(i32 { mag: 3785, sign: false });
    data.append(i32 { mag: 2554, sign: false });
    data.append(i32 { mag: 1056, sign: false });
    data.append(i32 { mag: 1490, sign: true });
    data.append(i32 { mag: 8761, sign: false });
    data.append(i32 { mag: 2683, sign: false });
    data.append(i32 { mag: 511, sign: false });
    data.append(i32 { mag: 1186, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
