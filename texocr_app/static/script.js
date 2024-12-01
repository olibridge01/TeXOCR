
// Create mapping of token to color for highlighting
const tokenToColour = {};
const possibleTokens = Array.from({ length: 1000 }, (_, i) => i.toString());

possibleTokens.forEach((token) => {
    tokenToColour[token] = getRandomColour();
});

const dropArea = document.getElementById("drop-area");
// Handle drag and drop
dropArea.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropArea.classList.add("hover");
});

dropArea.addEventListener("dragleave", () => {
    dropArea.classList.remove("hover");
});

const loadingSpinner = document.getElementById("loading-spinner");
const dropMessage = document.querySelector(".drop-area-text");
// Show loading spinner while waiting for prediction
function displayLoadingSpinner() {
    loadingSpinner.style.display = "block";
    dropMessage.style.display = "none";
}

// Hide loading spinner when prediction is displayed
function hideLoadingSpinner() {
    loadingSpinner.style.display = "none";
    dropMessage.style.display = "flex";
}

// Global variable to store the uploaded file in case of regeneration
let uploadedFile = null; 

dropArea.addEventListener("drop", (e) => {
    e.preventDefault();
    dropArea.classList.remove("hover");
    const file = e.dataTransfer.files[0];

    // Save file variable to be used in regenButton.onclick
    const image = document.getElementById("image-preview");
    image.src = URL.createObjectURL(file);
    
    if (file && file.type.startsWith("image/")) {

        previewImage(file);
        uploadedFile = file;
        uploadImage(file);
        displayLoadingSpinner();
    }
});

const fileInput = document.getElementById("file-input");
// Handle file input (in case user clicks the "Upload" button)
fileInput.addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (file && file.type.startsWith("image/")) {
        previewImage(file);
        uploadImage(file);
    }
});

const imagePreview = document.getElementById("image-preview");
// Preview image before upload
function previewImage(file) {
    const reader = new FileReader();
    reader.onload = () => {
        imagePreview.src = reader.result;
        imagePreview.style.display = "block";
    };
    reader.readAsDataURL(file);
}

// Upload image and get LaTeX prediction
async function uploadImage(file) {
    const formData = new FormData();
    formData.append("image", file);

    try {
        const response = await fetch("/predict", {
            method: "POST",
            body: formData,
        });
        const data = await response.json();
        displayPrediction(data);
        hideLoadingSpinner();
    } catch (error) {
        console.error("Error uploading image:", error);
    }
}

// Variables for displaying prediction
const latexResult = document.getElementById("latex-result");
const latexDiv = document.getElementById("latex");
const tokenizedResult = document.getElementById("tokenized-result");
const tokensDiv = document.getElementById("tokens");
const renderedResult = document.getElementById("rendered-result");
const renderDiv = document.getElementById("rendered");
const tokensButton = document.getElementById("tokens-button");
const regenButton = document.getElementById("regen-button");

// Display the LaTeX and tokenized results
function displayPrediction(data) {

    const container = document.querySelector(".container");
    const initialHeight = container.offsetHeight;
    // Set height to current height to prevent jumping
    // container.style.height = `${container.scrollHeight}px`;

    // If first time an image has been rendered, change drop area text
    const dropMessage = document.getElementById("drop-msg");
    if (dropMessage.textContent === "Drop here") {
        dropMessage.textContent = "Drop another";
    }

    latexResult.style.display = "block";
    latexDiv.textContent = data.latex;

    const errorMessage = document.getElementById("error-msg");
    errorMessage.style.display = "none";


    renderDiv.innerHTML = "";
    renderedResult.style.display = "block";
    const renderedLatex = katex.renderToString(data.latex, {
        throwOnError: false,
        output: "html",
        font: "tex",
        displayMode: true,
    });

    if (renderedLatex.includes("katex-error")) {
        // If the LaTeX was not rendered, show an error message
        console.error("KaTeX did not render the LaTeX:", data.latex);
        errorMessage.style.display = "block";
    } else {
        renderDiv.innerHTML = renderedLatex;
        console.log("Rendered LaTeX:", renderedLatex);
    }
    

    tokenizedResult.style.display = "none";
    tokensDiv.innerHTML = generateTokensWithHighlighting(data.strtokens, data.tokens);

    tokensButton.style.display = "block";
    tokensButton.textContent = "See Tokens";
    tokensButton.onclick = () => toggleTokens(data);

    regenButton.style.display = "block";
    regenButton.textContent = "Regenerate";
    
    regenButton.onclick = () => {
        if (uploadedFile) {
            uploadImage(uploadedFile);
            displayLoadingSpinner();
        }
    }

}

function toggleTokens(data) {
    if (tokenizedResult.style.display === "none") {

        const container = document.querySelector(".container");

        // Show tokens and swap LaTeX
        tokenizedResult.style.display = "block";
        
        // Add the data.strtokens to latexDiv, but each one highlighted with the corresponding color from data.tokens
        latexDiv.textContent = "";  // Clear the original LaTeX
        latexDiv.innerHTML = generateLatexWithHighlighting(data.strtokens, data.tokens);
        

        tokensButton.textContent = "Hide Tokens";  // Change button text
        addHoverHighlighting();

    } else {

        latexDiv.textContent = data.latex;  // Show original LaTeX
        tokensButton.textContent = "See Tokens";  // Change button text
        tokenizedResult.style.display = "none";

    }
}

function generateTokensWithHighlighting(str_tokens, tokens) {
    // Generate tokens with highlighting
    let tokenHTML = "[";
    for (let i = 0; i < str_tokens.length; i++) {
        const token = tokens[i];
        const color = tokenToColour[token];
        tokenHTML += `<span class="token" style="--token-color: ${color}">${token}</span>`;
    }
    tokenHTML += "]";
    return tokenHTML;
}

function generateLatexWithHighlighting(str_tokens, tokens) {
    // Generate LaTeX with highlighting for each token
    let latex = "";
    for (let i = 0; i < str_tokens.length; i++) {
        const token = tokens[i];
        const color = tokenToColour[token];
        latex += `<span class="str-token" style="background-color: ${color}">${str_tokens[i]}</span>`;
    }
    return latex;
}

function addHoverHighlighting() {
    const tokens = document.querySelectorAll(".token");
    const latexText = document.querySelectorAll(".str-token");

    tokens.forEach((tokenElement, index) => {
        tokenElement.addEventListener("mouseenter", () => highlightMatching(index, true));
        tokenElement.addEventListener("mouseleave", () => highlightMatching(index, false));
    });

    latexText.forEach((textElement, index) => {
        textElement.addEventListener("mouseenter", () => highlightMatching(index, true));
        textElement.addEventListener("mouseleave", () => highlightMatching(index, false));
    });

}

function highlightMatching(index, isHighlighted) {

    // Get colors for all tokens and text elements
    const tokenElements = document.querySelectorAll(".token");
    const textElements = document.querySelectorAll(".str-token");

    const color = tokenElements[index].style.getPropertyValue("--token-color");

    // Highlight corresponding token and LaTeX text
    if (isHighlighted) {

        // Set all textElements with no color except the one with the same color as the token
        for (let i = 0; i < textElements.length; i++) {
            textElements[i].style.backgroundColor = "";
        }

        tokenElements[index].style.backgroundColor = color;
        textElements[index].style.backgroundColor = color;
    } else {
        tokenElements[index].style.backgroundColor = "";

        // Set all textElements back to their original color
        for (let i = 0; i < textElements.length; i++) {
            textElements[i].style.backgroundColor = tokenElements[i].style.getPropertyValue("--token-color");
        }
            
    }


}


function getRandomColour() {
    "use strict";

    function randomInt(min, max) {
        return Math.floor(Math.random() * (max - min + 1)) + min;
    }

    const h = randomInt(0, 360);
    const s = randomInt(42, 98);
    const l = randomInt(75, 85);

    return `hsl(${h},${s}%,${l}%)`;
}

const copyButton = document.getElementById("copyButton");
const copiedMessage = document.getElementById("copied-msg");

// Copy LaTeX to clipboard if copyButton is clicked
// Make textcontent change to tick font awesome icon
copyButton.addEventListener("click", () => {
    const text = latexDiv.textContent;
    const icon = document.getElementById("copy-icon");

    navigator.clipboard.writeText(text).then(() => {

        // Smoothly change the icon to a tick for 1 second
        icon.classList.remove("fa-copy");
        icon.classList.add("fa-check");

        // Show the "Copied!" message for 2 seconds
        copiedMessage.style.display = "block";


        setTimeout(() => {
            icon.classList.remove("fa-check");
            icon.classList.add("fa-copy");

            copiedMessage.style.display = "none";
        }, 2000);
        
    });
});

function renderBackgroundEquations() {
    fetch("/static/bg_eqs.txt")
        .then((response) => {
            if (!response.ok) throw new Error("Failed to load equations file");
            return response.text();
        })
        .then((text) => {
            const equations = text
                .split("\n")
                .filter((line) => line.trim() !== ""); // Remove empty lines

            const wall = document.getElementById("equation-wall");

            // Add equations to the wall
            equations.forEach((equation) => {
                const div = document.createElement("div");
                div.classList.add("equation");

                try {
                    katex.render(equation, div, { throwOnError: false });
                } catch (err) {
                    console.error("KaTeX render error:", err);
                    console.log("Equation:", equation);
                }
                wall.appendChild(div);
            });
        })
        .catch((error) => console.error("Error rendering equations:", error));
}
document.addEventListener("DOMContentLoaded", renderBackgroundEquations);


// Function to render \TeX logo in banner then add 'OCR' to it
function renderLogoTex() {
    const logo = document.getElementById("title");
    katex.render("\\TeX", logo, { throwOnError: false });

    const ocr = document.createElement("span");
    ocr.textContent = "OCR";
    logo.appendChild(ocr);
}
document.addEventListener("DOMContentLoaded", renderLogoTex);