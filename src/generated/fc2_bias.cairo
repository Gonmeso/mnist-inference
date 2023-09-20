use array::ArrayTrait;
use orion::operators::tensor::{TensorTrait, Tensor, I32Tensor};
use orion::numbers::i32;


fn fc2_bias() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(10);
    let mut data = ArrayTrait::<i32>::new();
    data.append(i32 { mag: 466, sign: true });
    data.append(i32 { mag: 869, sign: false });
    data.append(i32 { mag: 27, sign: true });
    data.append(i32 { mag: 1015, sign: true });
    data.append(i32 { mag: 73, sign: false });
    data.append(i32 { mag: 2138, sign: false });
    data.append(i32 { mag: 276, sign: true });
    data.append(i32 { mag: 734, sign: true });
    data.append(i32 { mag: 1335, sign: true });
    data.append(i32 { mag: 418, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
