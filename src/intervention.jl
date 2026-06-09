"Hard intervention: set the within-unit treatment to the point mass at `a_star`."
struct Hard
    a_star::Float64
end

"Soft intervention: mix mass `ε` at `a_star` into the within-unit treatment distribution."
struct Soft
    a_star::Float64
    ε::Float64
end

default_baseline(intv::Hard) = Hard(0.0)
default_baseline(intv::Soft) = Soft(intv.a_star, 0.0)

Base.:(==)(x::Hard, y::Hard) = x.a_star == y.a_star
Base.:(==)(x::Soft, y::Soft) = x.a_star == y.a_star && x.ε == y.ε
