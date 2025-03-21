document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('fraudForm');
    const amount = document.getElementById('amount');
    const oldbalanceOrg = document.getElementById('oldbalanceOrg');
    const newbalanceOrig = document.getElementById('newbalanceOrig');
    const oldbalanceDest = document.getElementById('oldbalanceDest');
    const newbalanceDest = document.getElementById('newbalanceDest');

    // Validate amount is not negative
    amount.addEventListener('input', function() {
        if (this.value < 0) {
            this.value = 0;
        }
    });

    // Auto-calculate new balance based on old balance and amount
    amount.addEventListener('change', function() {
        if (oldbalanceOrg.value) {
            const newBalance = parseFloat(oldbalanceOrg.value) - parseFloat(this.value);
            newbalanceOrig.value = Math.max(0, newBalance).toFixed(2);
        }
    });

    oldbalanceOrg.addEventListener('change', function() {
        if (amount.value) {
            const newBalance = parseFloat(this.value) - parseFloat(amount.value);
            newbalanceOrig.value = Math.max(0, newBalance).toFixed(2);
        }
    });

    // Form validation before submit
    form.addEventListener('submit', function(e) {
        const amount = parseFloat(document.getElementById('amount').value);
        const oldBalanceOrg = parseFloat(document.getElementById('oldbalanceOrg').value);
        const newBalanceOrig = parseFloat(document.getElementById('newbalanceOrig').value);
        const oldBalanceDest = parseFloat(document.getElementById('oldbalanceDest').value);
        const newBalanceDest = parseFloat(document.getElementById('newbalanceDest').value);

        let isValid = true;
        let errorMessage = '';

        // Check if amount is greater than old balance
        if (amount > oldBalanceOrg) {
            isValid = false;
            errorMessage = 'Transaction amount cannot be greater than the origin account balance.';
        }

        // Validate balance changes
        if (oldBalanceOrg - amount !== newBalanceOrig) {
            isValid = false;
            errorMessage = 'New origin balance does not match the transaction amount.';
        }

        if (!isValid) {
            e.preventDefault();
            alert(errorMessage);
        }
    });
}); 