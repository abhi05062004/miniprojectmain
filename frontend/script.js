// Search Food Function
function searchFood() {
    let food = document.getElementById("foodInput").value;
    if (food.trim() === "") {
        alert("Please enter a food item.");
        return;
    }
    document.getElementById("results").innerHTML = `Searching for substitutes for <b>${food}</b>...`;
}

// Upload Image Function
function uploadImage() {
    let imageFile = document.getElementById("imageUpload").files[0];
    if (!imageFile) {
        alert("Please upload an image.");
        return;
    }
    document.getElementById("results").innerHTML = `Image <b>${imageFile.name}</b> uploaded successfully.`;
}

// Submit Data Function
async function submitData() {
    let food = document.getElementById("foodInput").value;
    let imageFile = document.getElementById("imageUpload").files[0];
    let resultsDiv = document.getElementById("results");

    if (food.trim() === "" && !imageFile) {
        alert("Please enter a food item or upload an image.");
        return;
    }

    resultsDiv.innerHTML = "Processing request...";

    try {
        let response;
        if (imageFile) {
            let formData = new FormData();
            formData.append("file", imageFile);
            response = await fetch("http://localhost:5000/predict", {
                method: "POST",
                body: formData
            });
        } else {
            response = await fetch("http://localhost:5000/substitute", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ ingredient: food })
            });
        }

        let result = await response.json();
        if (result.error) {
            resultsDiv.innerHTML = `<span style='color: red;'>Error: ${result.error}</span>`;
        } else {
            resultsDiv.innerHTML = `Substitute for <b>${result.ingredient}</b>: <b>${result.substitute}</b>`;
        }
    } catch (error) {
        resultsDiv.innerHTML = `<span style='color: red;'>Error processing request.</span>`;
    }
}
