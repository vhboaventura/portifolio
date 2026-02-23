import numpy as np
import matplotlib.pyplot as plt
import numpy as np

def solver_jacobi(Nx, k0, phi, Lx, p_ini, A, p_e, visc,dt):
    # Tempo e malha
    tempo_total=364*86400
    n_steps=200
    dx = Lx / Nx
    #dt = tempo_total / n_steps  # fixar o tempo total da simulação
    x = np.linspace(dx/2, Lx - dx/2, Nx)  # centros dos volumes

    # Propriedades físicas
    c0 = 6e-7         # compressibilidade [1/Pa]
    k = np.full(Nx, k0)  # permeabilidade constante
    k = k0 + 1e-10*x
    T = np.zeros(Nx - 1)

    for i in range(Nx - 1):
        keq = 2 * k[i] * k[i+1] / (k[i] + k[i+1])  # média harmônica
        T[i] = (keq / visc) * (A / dx)

    beta = phi * c0 * dx * A / dt

    # Inicialização da pressão
    p = np.full(Nx, p_ini)
    p_all = [p.copy()]
    times = [0.0]

    tol = 1e-4
    max_iter = 10000

    for step in range(1, n_steps + 1):
        p_old = p.copy()
        it = 0

        while it < max_iter:
            p_new = p.copy()
            p_new[0] = p_e  # pressão prescrita

            erro = 0.0
            for i in range(1, Nx - 1):
                aW = T[i - 1]
                aE = T[i]
                aP = aW + aE + beta
                b = beta * p_old[i]
                p_calc = (aW * p_new[i - 1] + aE * p[i + 1] + b) / aP
                erro = max(erro, abs(p_calc - p_new[i]))
                p_new[i] = p_calc

            # Condição de Neumann (fluxo nulo na saída)
            p_new[-1] = p_new[-2]

            if erro < tol:
                break

            p = p_new
            it += 1

        p = p_new.copy()
        p_all.append(p.copy())
        times.append(step * dt)

        # Progresso
        print(f'\rProgresso: {(step / n_steps) * 100:.2f}%', end='')

    print('\nSimulação finalizada!')
    return np.array(p_all), np.array(times), x

# ======================
# ROTINA DE REFINAMENTO
# ======================
k0 = 10e-14
phi = 0.25
p_ini = 45000e3
p_e = 70000e3
Lx = 5000
Ly = 40
Lz = 10
A = Ly * Lz
visc = 1.2e-3      # Pa.s

#Plot para variação de Nx
resolucoes = [10, 50, 100, 200, 400]
cores = ['blue', 'orange', 'green', 'red', 'purple']

'''plt.figure(figsize=(10, 6))

for Nx, cor in zip(resolucoes, cores):
    p_all, times, x = solver_jacobi(Nx, k0, phi, Lx, p_ini, A, p_e, visc=1.2e-3)
    plt.plot(x, p_all[-1] / 1e6, label=str(Nx), color=cor)

plt.xlabel('x (m)')
plt.ylabel('Pressão (MPa)')
plt.title('Distribuição de Pressão ao longo do espaço')
plt.legend(title="Nx")
plt.grid(True)
plt.tight_layout()
plt.show()'''

#Plot - Pressão ao longo do espaço variando visc
'''p1, times, eixo1 = solver_jacobi(200, k0, phi, Lx, p_ini, A, p_e, 1.2e-3)
p2, times, eixo2 = solver_jacobi(200, k0, phi, Lx, p_ini, A, p_e, 1.2e-2)
p3, times, eixo3 = solver_jacobi(200, k0, phi, Lx, p_ini, A, p_e, 1.2e-4)
plt.figure(figsize=(8, 5))
plt.plot(eixo2, p2[-1]/10e6, label = 'visc = 1.2e-2 Pa.s')
plt.plot(eixo1, p1[-1]/10e6, label = 'visc = 1.2e-3 Pa.s')
plt.plot(eixo3, p3[-1]/10e6, label = 'visc = 1.2e-4 Pa.s')
plt.xlabel('x (m)')
plt.ylabel('Pressão (MPa)')
plt.title('Distribuição de Pressão ao longo do espaço')
plt.legend()
plt.grid()
plt.show()'''

#Plot - Pressão ao longo do espaço variando phi
'''p1, times, eixo1 = solver_jacobi(200, k0, 0.25, Lx, p_ini, A, p_e, visc)
p2, times, eixo2 = solver_jacobi(200, k0, 0.1, Lx, p_ini, A, p_e, visc)
p3, times, eixo3 = solver_jacobi(200, k0, 0.5, Lx, p_ini, A, p_e, visc)
plt.figure(figsize=(8, 5))
plt.plot(eixo2, p2[-1]/10e6, label = 'phi = 0.1')
plt.plot(eixo1, p1[-1]/10e6, label = 'phi = 0.25')
plt.plot(eixo3, p3[-1]/10e6, label = 'phi = 0.5')
plt.xlabel('x (m)')
plt.ylabel('Pressão (MPa)')
plt.title('Distribuição de Pressão ao longo do espaço')
plt.legend()
plt.grid()
plt.show()'''

#Plot - Pressão ao longo do espaço variando pressão de entrada
'''p1, times, eixo1 = solver_jacobi(200, k0, phi, Lx, p_ini, A, 70e6 , visc)
p2, times, eixo2 = solver_jacobi(200, k0, phi, Lx, p_ini, A, 85e6, visc)
p3, times, eixo3 = solver_jacobi(200, k0, phi, Lx, p_ini, A, 100e6, visc)
plt.figure(figsize=(8, 5))
plt.plot(eixo1, p1[-1]/1e6, label = 'p_e = 70 MPa')
plt.plot(eixo2, p2[-1]/1e6, label = 'p_e = 85 MPa')
plt.plot(eixo3, p3[-1]/1e6, label = 'p_e = 100 MPa')
plt.xlabel('x (m)')
plt.ylabel('Pressão (MPa)')
plt.title('Distribuição de Pressão ao longo do espaço')
plt.legend()
plt.grid()
plt.show()'''

#Plot - Pressão ao longo do espaço variando pressão de entrada
'''p1, times, eixo1 = solver_jacobi(200, k0, phi, Lx, p_ini, A, 70e6 , visc)
p2, times, eixo2 = solver_jacobi(200, k0, phi, Lx, p_ini, A, 85e6, visc)
p3, times, eixo3 = solver_jacobi(200, k0, phi, Lx, p_ini, A, 100e6, visc)
plt.figure(figsize=(8, 5))
plt.plot(eixo1, p1[-1]/1e6, label = 'p_e = 70 MPa')
plt.plot(eixo2, p2[-1]/1e6, label = 'p_e = 85 MPa')
plt.plot(eixo3, p3[-1]/1e6, label = 'p_e = 100 MPa')
plt.xlabel('x (m)')
plt.ylabel('Pressão (MPa)')
plt.title('Distribuição de Pressão ao longo do espaço')
plt.legend()
plt.grid()
plt.show()'''

#Plot - Pressão ao longo do espaço variando área
'''p1, times, eixo1 = solver_jacobi(200, k0, phi, Lx, p_ini, A, p_e , visc)
p2, times, eixo2 = solver_jacobi(200, k0, phi, Lx, p_ini, 800, p_e, visc)
p3, times, eixo3 = solver_jacobi(200, k0, phi, Lx, p_ini, 1600, p_e, visc)
plt.figure(figsize=(8, 5))
plt.plot(eixo1, p1[-1]/1e6, label = 'A = 400 m²')
plt.plot(eixo2, p2[-1]/1e6, label = 'A = 800 m²')
plt.plot(eixo3, p3[-1]/1e6, label = 'A = 1600 m²')
plt.xlabel('x (m)')
plt.ylabel('Pressão (MPa)')
plt.title('Distribuição de Pressão ao longo do espaço')
plt.legend()
plt.grid()
plt.show()'''

# Teste variando o Lx e plotando pressão vs posição real (x já depende de Lx)
'''Lx_vals = [2000, 5000, 8000]
cores = ['blue', 'green', 'red']

plt.figure(figsize=(8, 5))

for Lx, cor in zip(Lx_vals, cores):
    p_all, times, x = solver_jacobi(200, k0, phi, Lx, p_ini, A, p_e, visc)
    plt.plot(x, p_all[-1] / 1e6, label=f'Lx = {Lx} m', color=cor)

plt.xlabel('x (m)')
plt.ylabel('Pressão (MPa)')
plt.title('Distribuição de Pressão ao longo do espaço')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()'''

# Valores de dt (em segundos)
dt_vals = [2 * 86400, 1 * 86400, 0.5 * 86400]  # 2 dias, 1 dia, 0.5 dia
cores = ['blue', 'green', 'red']

plt.figure(figsize=(8, 5))

for dt, cor in zip(dt_vals, cores):
    p_all, times, x = solver_jacobi(200, k0, phi, Lx, p_ini, A, p_e, visc, dt)
    plt.plot(x, p_all[-1] / 1e6, label=f'dt = {dt/86400:.1f} dias', color=cor)

plt.xlabel('x (m)')
plt.ylabel('Pressão (MPa)')
plt.title('Distribuição de Pressão ao longo do espaço variando dt')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


