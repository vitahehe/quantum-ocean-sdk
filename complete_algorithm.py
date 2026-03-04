"""Top‑level entry point that ties together Hamiltonian generation and QUBO solvers."""
import os

def main():
    import numpy as np
    from trackhhl.toy.simple_generator import SimpleGenerator
    from trackhhl.toy.simple_generator import SimpleDetectorGeometry
    from Hamiltonian import generate_hamiltonian_optimizedPAR
    from QUBO_solvers import qubosolverSA, qubosolverQA, qubosolverHr, brute_force

    # build simple event
    N_MODULES = 3
    LX = float("+inf")
    LY = float("+inf")
    Z_SPACING = 1.0
    detector = SimpleDetectorGeometry(
        module_id=list(range(N_MODULES)),
        lx=[LX] * N_MODULES,
        ly=[LY] * N_MODULES,
        z=[i + Z_SPACING for i in range(N_MODULES)],
    )
    generator = SimpleGenerator(detector_geometry=detector, theta_max=np.pi/6)
    event = generator.generate_event(2)

    params = {"lambda": 1.0, "alpha": 0.5, "beta": 0.1}
    A, b, true_sol = generate_hamiltonian_optimizedPAR(event, params)
    print("true solution", true_sol)

    # run available solvers
    sa = qubosolverSA(A, b)
    qa = qubosolverQA(A, b, token=os.getenv('DWAVE_API_TOKEN'))
    hr = qubosolverHr(A, b)
    brute = brute_force(A, b)
    print(sa, qa, hr, brute)

if __name__ == "__main__":
    main()
