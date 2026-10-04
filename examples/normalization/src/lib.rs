use cubecl::{Device, prelude::*};

#[cube(launch)]
fn norm_test<F: Float, N: Size>(
    input: &[Vector<F, N>],
    output_a: &mut [Vector<F, N>],
    output_b: &mut [Vector<F, N>],
) {
    if ABSOLUTE_POS < input.len() {
        output_a[ABSOLUTE_POS] = input[ABSOLUTE_POS].normalize();
        output_b[ABSOLUTE_POS] =
            input[ABSOLUTE_POS] / Vector::new(input[ABSOLUTE_POS].magnitude());
    }
}

pub fn launch(device: &Device) {
    let client = device.client();
    let input = Buffer::create(&client, &[-1f32, 0., 1., 5.]);
    let output_a = Buffer::<f32>::empty(&client, input.len());
    let output_b = Buffer::<f32>::empty(&client, input.len());

    norm_test::launch::<f32>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(1),
        4,
        (&input).into(),
        (&output_a).into(),
        (&output_b).into(),
    );

    let output = output_a.read(&client).unwrap();

    println!(
        "Executed normalize with runtime {:?} => {output:?}",
        client.name()
    );

    let output = output_b.read(&client).unwrap();

    println!(
        "Executed normalize using magnitude with runtime {:?} => {output:?}",
        client.name()
    );
}
