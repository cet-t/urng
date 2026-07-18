#[macro_export]
macro_rules! impl_choice {
    ($bits:expr) => {
        ::paste::paste! {
            pub trait [<Choice $bits>]: [<Rng $bits>] {
                /// Returns a random element from a slice.
                #[inline(always)]
                fn choice<'a, T>(&mut self, choices: &'a [T]) -> &'a T {
                    let index = self.randi(0, choices.len() as [<i $bits>] - 1);
                    &choices[index as usize]
                }
            }

            impl<T: [<Rng $bits>] + ?Sized> [<Choice $bits>] for T {}
        }
    };
}
