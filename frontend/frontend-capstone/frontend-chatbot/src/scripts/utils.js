document.addEventListener("DOMContentLoaded", function () {
    const textarea = document.querySelector("main textarea");
    const initialHeight = textarea.clientHeight;
    let isExpanded = false;

    textarea.addEventListener("input", function () {
        if (this.scrollHeight > this.clientHeight && !isExpanded) {
            this.style.height = `${this.clientHeight * 2}px`;
            isExpanded = true; 
        } else if (this.value.trim() === "") {
            this.style.height = `${initialHeight}px`;
            isExpanded = false; 
        }
    });
});
