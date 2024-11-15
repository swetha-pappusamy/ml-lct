document.addEventListener('DOMContentLoaded', () => {
    const categoriesElement = document.getElementById('categories');
    const productsElement = document.getElementById('products');
    const categoryListElement = document.getElementById('category-list');
    const productListElement = document.getElementById('product-list');
    const backButton = document.getElementById('back-button');

    fetch('data.json')
        .then(response => response.json())
        .then(data => {
            const categories = Object.keys(data);
            categories.forEach(category => {
                const li = document.createElement('li');
                li.textContent = category;
                li.addEventListener('click', () => {
                    showProducts(data[category]);
                });
                categoriesElement.appendChild(li);
            });
        });

    backButton.addEventListener('click', () => {
        categoryListElement.classList.remove('hidden');
        productListElement.classList.add('hidden');
    });

    function showProducts(products) {
        productsElement.innerHTML = '';
        products.forEach(product => {
            const li = document.createElement('li');
            li.textContent = `${product.product} - ${product.buyers} people bought this`;
            productsElement.appendChild(li);
        });
        categoryListElement.classList.add('hidden');
        productListElement.classList.remove('hidden');
    }
});