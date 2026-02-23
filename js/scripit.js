// Função para o menu de navegação móvel
function setupMobileMenu() {
    const nav = document.querySelector('.nav');
    const menuToggle = document.createElement('div');
    menuToggle.classList.add('menu-toggle');
    menuToggle.innerHTML = '<i class="bx bx-menu"></i>'; // Ícone de menu
    document.querySelector('header').appendChild(menuToggle);

    // Alternar a visibilidade do menu ao clicar no ícone
    menuToggle.addEventListener('click', () => {
        nav.classList.toggle('active');
        menuToggle.querySelector('i').classList.toggle('bx-x'); // Alternar ícone de menu e X
    });

    // Fechar o menu ao clicar em um link
    nav.querySelectorAll('a').forEach(link => {
        link.addEventListener('click', () => {
            nav.classList.remove('active');
            menuToggle.querySelector('i').classList.remove('bx-x');
        });
    });
}

// Função para scroll suave
function setupSmoothScroll() {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault(); // Previne o comportamento padrão
            const targetId = this.getAttribute('href'); // Pega o ID da seção
            const targetSection = document.querySelector(targetId);

            if (targetSection) {
                // Scroll suave até a seção
                targetSection.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
}

// Função para animação de scroll (reveal)
function setupScrollReveal() {
    const sections = document.querySelectorAll('section, article');

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible'); // Adiciona classe para animação
            }
        });
    }, {
        threshold: 0.1 // Define o limite de visibilidade
    });

    sections.forEach(section => {
        observer.observe(section); // Observa cada seção
    });
}

// Inicializa todas as funções
document.addEventListener('DOMContentLoaded', () => {
    setupMobileMenu();
    setupSmoothScroll();
    setupScrollReveal();
});