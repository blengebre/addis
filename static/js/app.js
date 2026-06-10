document.addEventListener('DOMContentLoaded', () => {
    const sourceText = document.getElementById('sourceText');
    const targetText = document.getElementById('targetText');
    const translateBtn = document.getElementById('translateBtn');
    const clearBtn = document.getElementById('clearBtn');
    const copyBtn = document.getElementById('copyBtn');
    const charCount = document.getElementById('charCount');
    const toast = document.getElementById('toast');

    // Update character count
    sourceText.addEventListener('input', () => {
        const length = sourceText.value.length;
        charCount.textContent = `${length} character${length !== 1 ? 's' : ''}`;
    });

    // Clear input
    clearBtn.addEventListener('click', () => {
        sourceText.value = '';
        targetText.innerHTML = '<span class="placeholder-text">Translation will appear here...</span>';
        targetText.classList.remove('error-text');
        charCount.textContent = '0 characters';
        sourceText.focus();
    });

    // Copy to clipboard
    copyBtn.addEventListener('click', () => {
        const textToCopy = targetText.textContent;
        // Don't copy placeholder or empty text
        if (textToCopy === 'Translation will appear here...' || !textToCopy.trim()) {
            return;
        }

        navigator.clipboard.writeText(textToCopy).then(() => {
            showToast();
        }).catch(err => {
            console.error('Failed to copy text: ', err);
        });
    });

    function showToast() {
        toast.classList.add('show');
        setTimeout(() => {
            toast.classList.remove('show');
        }, 2000);
    }

    // Handle translation
    translateBtn.addEventListener('click', async () => {
        const text = sourceText.value.trim();
        
        if (!text) {
            sourceText.focus();
            return;
        }

        // UI updates for loading state
        translateBtn.classList.add('translating');
        translateBtn.disabled = true;
        
        try {
            const response = await fetch('/api/translate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text: text })
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Translation failed');
            }

            // Display result
            targetText.textContent = data.translation;
            targetText.classList.remove('placeholder-text', 'error-text');
            
        } catch (error) {
            console.error('Translation error:', error);
            targetText.innerHTML = `<span style="color: #ef4444;">Error: ${error.message}. Please try again.</span>`;
        } finally {
            // Restore button state
            translateBtn.classList.remove('translating');
            translateBtn.disabled = false;
        }
    });

    // Allow Ctrl+Enter to submit
    sourceText.addEventListener('keydown', (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            translateBtn.click();
        }
    });
});
