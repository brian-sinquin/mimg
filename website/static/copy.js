function copyCommand(button) {
  const commandContainer = button.parentElement;
  const code = commandContainer.querySelector('code');
  const text = 'mimg lena.png ' + code.textContent.trim() + ' -o output.png';

  // Store original content
  const originalHTML = button.innerHTML;

  navigator.clipboard.writeText(text).then(() => {
    // Change to success state with checkmark icon
    button.innerHTML = '<svg class="copy-icon" width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z"/></svg>Copied';
    button.classList.add('copied');
    button.style.background = 'rgba(76, 175, 80, 0.8)';
    button.style.pointerEvents = 'none';

    setTimeout(() => {
      button.innerHTML = originalHTML;
      button.classList.remove('copied');
      button.style.background = '';
      button.style.pointerEvents = 'auto';
    }, 1500);
  }).catch(err => {
    console.error('Failed to copy:', err);

    // Show error state with X icon
    button.innerHTML = '<svg class="copy-icon" width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/></svg>Error';
    button.classList.add('error');
    button.style.background = 'rgba(244, 67, 54, 0.8)';
    button.style.pointerEvents = 'none';

    setTimeout(() => {
      button.innerHTML = originalHTML;
      button.classList.remove('error');
      button.style.background = '';
      button.style.pointerEvents = 'auto';
    }, 1500);
  });
}
