use prettytable::*;
use rand::{thread_rng, Rng};

const P: f64 = 0.95;
const E: f64 = 0.1;
const L: usize = 10;

fn harmonic_mean(noisy_func: &[f64], k: usize, alpha: &[f64], r: usize) -> f64 {
    let mut mean = 0f64;
    let m = r / 2;
    if k < m {
        for i in 0..=(k + m) {
            mean += alpha[i] / noisy_func[i];
        }
    } else if k + m > noisy_func.len() - 1 {
        for i in (k - m)..k {
            mean += alpha[i + m - k] / noisy_func[i];
        }
    } else {
        for i in (k - m)..=(k + m) {
            mean += alpha[i + m - k] / noisy_func[i];
        }
    }
    1f64 / mean
}

fn get_alpha(r: usize) -> Vec<f64> {
    let mut alpha = vec![0f64; r];
    let m = (r + 1) / 2 - 1;
    alpha[m] = thread_rng().gen_range(0f64..=1f64);
    let mut alpha_sum = alpha[m];
    for i in 1..=m {
        alpha[m - i] = 0.5f64 * thread_rng().gen_range(0f64..=(1f64 - alpha_sum));
        alpha[m + i] = alpha[m - i];
        alpha_sum += 2f64 * alpha[m - i];
    }
    alpha[r - 1] = (1f64 - alpha_sum) / 2f64;
    alpha[0] = alpha[r - 1];
    alpha
}

fn find_alpha(noisy_func: &[f64], r: usize, lambda: f64, range: (f64, f64)) -> Vec<f64> {
    let mut alpha = get_alpha(r);
    let mut crit = int_crit(noisy_func, &alpha, r, lambda);
    for _ in 0..((1f64 - P).log(std::f64::consts::E)
        / (1f64 - (E / (range.1 - range.0))).log(std::f64::consts::E))
    .round() as usize
    {
        let new_alpha = get_alpha(r);
        let new_crit = int_crit(noisy_func, &alpha, r, lambda);
        if new_crit < crit {
            alpha = new_alpha;
            crit = new_crit;
        }
    }
    alpha
}

fn noise_crit(noisy_func: &[f64], alpha: &[f64], r: usize) -> f64 {
    let mut omega = 0f64;
    for i in 1..noisy_func.len() {
        let first = harmonic_mean(noisy_func, i - 1, alpha, r);
        let second = harmonic_mean(noisy_func, i, alpha, r);
        omega += (second - first).powi(2);
    }
    omega
}

fn divergence_crit(noisy_func: &[f64], alpha: &[f64], r: usize) -> f64 {
    let mut delta = 0f64;
    for i in 0..noisy_func.len() {
        let mut temp = harmonic_mean(noisy_func, i, alpha, r);
        temp -= noisy_func[i];
        delta += temp.powi(2);
    }
    delta / (noisy_func.len() as f64)
}

fn int_crit(noisy_func: &[f64], alpha: &[f64], r: usize, lambda: f64) -> f64 {
    lambda * noise_crit(noisy_func, alpha, r)
        + (1f64 - lambda) * divergence_crit(noisy_func, alpha, r)
}

fn distance_noise(noisy_func: &[f64], alpha: &[f64], r: usize) -> f64 {
    (noise_crit(noisy_func, alpha, r).powi(2) + divergence_crit(noisy_func, alpha, r).powi(2))
        .sqrt()
}

pub fn filter(noisy_func: &[f64], r: usize, range: (f64, f64)) -> (Vec<f64>, Vec<(f64, f64)>) {
    let mut min_dist: Option<f64> = None;
    let mut best_lambda = None;
    let mut best_alpha = None;
    let mut best_delta = None;
    let mut best_omega = None;
    let mut best_crit = None;
    let mut table = Table::new();
    let mut coeffs_vector = Vec::<(f64, f64)>::with_capacity(L);
    table.add_row(row!["λ", "Distance", "ɑ", "⍵", "Δ"]);
    for i in 0..L {
        let lambda = (i as f64) / (L as f64);
        let alpha = find_alpha(noisy_func, r, lambda, range);
        let dist = distance_noise(noisy_func, &alpha, r);
        let omega = noise_crit(noisy_func, &alpha, r);
        let delta = divergence_crit(noisy_func, &alpha, r);
        let mut alpha_string = String::new();
        alpha_string.push_str("[ ");
        for i in alpha.iter().enumerate() {
            alpha_string.push_str(i.1.to_string().as_str());
            if i.0 < alpha.len() - 1 {
                alpha_string.push_str(", ")
            } else {
                alpha_string.push_str(" ]");
            }
        }
        table.add_row(row![lambda, dist, alpha_string, omega, delta]);
        coeffs_vector.push((omega.clone(), delta.clone()));
        if min_dist.is_none() || *min_dist.as_ref().unwrap() > dist {
            min_dist = Some(dist);
            best_crit = Some(int_crit(noisy_func, &alpha, r, lambda));
            best_lambda = Some(lambda);
            best_alpha = Some(alpha);
            best_omega = Some(omega);
            best_delta = Some(delta);
        }
    }
    table.printstd();
    println!("\n\n");
    let mut secondary = Table::new();
    secondary.add_row(row!["λ*", "J", "⍵", "Δ"]);
    secondary.add_row(row![
        best_lambda.unwrap(),
        best_crit.unwrap(),
        best_omega.unwrap(),
        best_delta.unwrap()
    ]);
    secondary.printstd();
    let mut filtered = Vec::<f64>::with_capacity(noisy_func.len());
    for i in 0..noisy_func.len() {
        filtered.push(harmonic_mean(
            noisy_func,
            i,
            best_alpha.as_ref().unwrap(),
            r,
        ));
    }
    (filtered, coeffs_vector)
}
