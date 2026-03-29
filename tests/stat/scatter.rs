use anyhow::Result;
use plotters::prelude::*;

/// Render a lag-1 scatter plot: each point is (x[i], x[i+1]) for consecutive outputs.
///
/// Saves to `path` as a PNG image (640×640 pixels).
pub fn plot(name: &str, rng_fn: &mut dyn FnMut() -> f64, n: usize, path: &str) -> Result<()> {
    let root = BitMapBackend::new(path, (640, 640)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(name, ("sans-serif", 18).into_font())
        .margin(20)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0f64..1f64, 0f64..1f64)?;

    chart
        .configure_mesh()
        .x_labels(5)
        .y_labels(5)
        .disable_mesh()
        .draw()?;

    let mut points = Vec::with_capacity(n);
    let mut prev = rng_fn();
    for _ in 0..n {
        let cur = rng_fn();
        points.push((prev, cur));
        prev = cur;
    }

    chart
        .draw_series(
            points
                .iter()
                .map(|&(x, y)| Circle::new((x, y), 1, RGBColor(60, 120, 200).filled())),
        )?
        .label(name)
        .legend(|(x, y)| Circle::new((x, y), 4, RGBColor(60, 120, 200).filled()));

    root.present()?;
    println!("  scatter → {}", path);

    Ok(())
}
