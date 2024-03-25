
let selectedTileType = null; // Variable to keep track of the selected tile type
let paused = false;
let ckptFolder = null;
let currTileMap = null;
let isDragging = false; // Flag to track dragging state

// TODO: Let the environment specify this.
pathTypes = ['path_g', 'path_g', 'path_e']

// Initialize a map of edit coordinates to tile types for current edits
currEdits = {};

function animateEdit(event) {
    animateTile(currTileMap, event.target.coords, selectedTileType);
}

function queueEdit(event) {
    // if coords undefined, do nothing
    if (event.target.coords === undefined) {
        return;
    }
    currEdits[event.target.coords] = selectedTileType;
    console.log('queueEdit', event.target.coords, selectedTileType);
}

document.getElementById('pathmap').addEventListener('mousedown', function(event) {
    console.log('mousedown', event.target.id, event.target.tagName)
    if ((event.target.parentNode.id === 'tilemap' || event.target.parentNode.id === 'pathmap') && event.target.tagName === 'IMG') {
        console.log('mousedown', event.target.parentNode.id, event.target.tagName)
        isDragging = true;
        queueEdit(event);
        animateEdit(event);
        console.log('mousedown', event.target);
    }
});

document.getElementById('pathmap').addEventListener('mousemove', function(event) {
    if (isDragging && (event.target.parentNode.id === 'tilemap' || event.target.parentNode.id === 'pathmap') && event.target.tagName === 'IMG') {
        if (event.target.coords === undefined) {
            return;
        }
        queueEdit(event);
        animateEdit(event);
    }
});

document.addEventListener('mouseup', function() {
    isDragging = false;
    if (Object.keys(currEdits).length > 0) {
        sendEdits();
    }
});

function sendEdits() {
    fetch('/apply_edits', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({edits: currEdits}),
    })
    .then(response => response.json())
    .then(data =>
        displayData(data)
    )
    .catch(error => console.error('Error:', error));
    console.log('sendEdits', currEdits);
    currEdits = {};
}

// Prevent the default drag behavior for images within the tilemap
document.getElementById('pathmap').addEventListener('dragstart', function(event) {
    event.preventDefault();
});

function placeTile(target) {
    if (target.tagName === 'IMG' && isDragging) { // Check if the target is an image (tile) and dragging is true
        const coords = target.coords;
        console.log('coords', coords);
        if (coords) {
            const [x, y] = coords.split(',').map(Number);
            if (currTileMap[y][x] != selectedTileType) {
                // Example of sending the message to Python (adjust based on your setup)
                fetch('/update_tile', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({coords: coords, tileType: selectedTileType}),
                })
                .then(response => response.json())
                .then(data =>
                    displayData(data)
                )
                .catch(error => console.error('Error:', error));
            
            }
        }
    }
}


// When a new folder is selected, get the string
document.getElementById('folderSelect').addEventListener('change', (event) => {
    ckpt_folder = event.target.value;
});

document.getElementById('tileCheckboxes').addEventListener('change', (event) => {
    if (event.target.type === 'checkbox') {
        // Update selectedTileType based on the checked checkbox
        selectedTileType = event.target.id.replace('tile-', '');
        // Ensure only one checkbox is checked at a time
        document.querySelectorAll('#tileCheckboxes input[type="checkbox"]').forEach((checkbox) => {
            if (checkbox !== event.target) {
                checkbox.checked = false;
            }
        });
    }
});

document.getElementById('pauseButton').addEventListener('click', function() {
    paused = !paused; // Toggle the pause state
    // Change text to reflect the current state
    this.textContent = paused ? 'Resume' : 'Pause';
    if (!paused) {
        tickEnv(); // Resume ticking if unpaused
    }
});

// document.getElementById('tilemap').addEventListener('click', (event) => {
//     // Assuming each tile has a data attribute like data-coords="x,y"
//     const coords = event.target.coords;
//     // console.log('coords', event.target.coords);
//     if (coords && selectedTileType !== null) {
//         // Retrieve the tile name from the selectedTileType
//         console.log(`Tile clicked at coords: ${coords}, Selected tile type: ${selectedTileType}`);

//         // Example of sending the message to Python (adjust based on your setup)
//         fetch('/update_tile', {
//             method: 'POST',
//             headers: {'Content-Type': 'application/json'},
//             body: JSON.stringify({coords: coords, tileType: selectedTileType}),
//         })
//         .then(response => response.json())
//         .then(data =>
//             displayData(data)
//         )
//         .catch(error => console.error('Error:', error));
//     }
// });

function resetEnv(event) {
    console.log("Environment resetting...");
    fetch('/reset_env', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({}),
    }).then(response => response.json())
    .then(data => {
        displayData(data);
    })
    .catch(error => console.error('Error:', error));
}

function tickEnv() {
    if (paused) {
        return; // Do nothing if paused
    }
    fetch('/tick')
        .then(response => response.json())
        .then(data => {
            displayData(data);
            tickEnv();
        })
        .catch(error => console.error('Error:', error));
}

function initEnv(event) {
    console.log("Environment initializing...");
    fetch('/init_env', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({}),
    }).then(response => response.json())
    .then(data => {
        displayData(data);
    })
    .catch(error => console.error('Error:', error));
    createTileCheckboxes();
}

function displayData(data) {
    displayTilemap(data.map);

    const pathMapElement = document.getElementById('pathmap');

    // Iterate through paths in data
    for (let i = 0; i < data.paths.length; i++) {
        // console.log(data.paths[i]);
        displayPath(data.paths[i], pathTypes[i]);
    }            
}

function initPathMap() {
    // Place transparent/no images on the pathmap
    const pathMapElement = document.getElementById('pathmap');
    pathMapElement.innerHTML = ''; // Clear previous tiles if any
    const totalColumns = currTileMap[0].length;
    const totalRows = currTileMap.length;
    // Create a transparent image using a data URL
    const transparentImageSrc = "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7";
    for (let i = 0; i < totalRows; i++) {
        for (let j = 0; j < totalColumns; j++) {
            const img = document.createElement('img');
            img.src = transparentImageSrc;
            img.style.width = '16px';
            img.style.height = '16px';
            img.coords = `${j},${i}`;
            pathMapElement.appendChild(img);
        }
    }
}

function displayPath(path, pathType) {
    const pathMapElement = document.getElementById('pathmap');
    const ctx = pathMapElement.getContext('2d');
    ctx.globalCompositeOperation = 'source-over'; // New shapes are drawn on top of existing content

    // Iterate through (x, y) in path
    for (let i = 0; i < path.length; i++) {
        const columnIndex = path[i][1];
        const rowIndex = path[i][0];
        if (columnIndex == -1) {
            return;
        }
        console.log(`displayPath: ${columnIndex}, ${rowIndex}, ${pathType}`)
        const img = new Image();
        img.src = `/static/tiles/${pathType}.png`; // Assumes tile images are named 'tile0.png', 'tile1.png', etc.
        img.onload = function() {
            ctx.drawImage(img, columnIndex * 16, rowIndex * 16, 16, 16); // Assumes each tile is 16x16 pixels
        }
    }
}

function displayTilemap(tileMap) {
    const pathMapElement = document.getElementById('pathmap');
    const ctx = pathMapElement.getContext('2d');
    ctx.globalCompositeOperation = 'source-over'; // New shapes are drawn on top of existing content

    // Set the size of the canvas to match the size of the tile map
    pathMapElement.width = tileMap[0].length * 16; // Assumes each tile is 16 pixels wide
    pathMapElement.height = tileMap.length * 16; // Assumes each tile is 16 pixels tall

    currTileMap = tileMap;

    console.log('displayTilemap height', tileMap.length, 'width', tileMap[0].length);
    tileMap.forEach((row, rowIndex) => {
        row.forEach((tile, columnIndex) => {
            const img = document.createElement('img');
            img.src = `/static/tiles/${tile}.png`; // Assumes tile images are named 'tile0.png', 'tile1.png', etc.
            img.onload = function() {
                ctx.drawImage(img, columnIndex * 16, rowIndex * 16, 16, 16); // Assumes each tile is 16x16 pixels
            }
        });
    });
}

function animateTile(tileMap, coords, tileType) {
    const [columnIndex, rowIndex] = coords.split(',').map(Number);
    const tileMapElement = document.getElementById('tilemap');
    const img = document.createElement('img');
    const ctx = tileMapElement.getContext('2d');
    img.src = `/static/tiles/${tileType}.png`; // Assumes tile images are named 'tile0.png', 'tile1.png', etc.
    img.onload = function() {
        ctx.drawImage(img, columnIndex * 16, rowIndex * 16, 16, 16); // Assumes each tile is 16x16 pixels
    }
}

function clearMap(tileMap) {
    fetch('/clear_map', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({}),
    }).then(response => response.json())
    .then(data => {
        displayData(data);
    })
    .catch(error => console.error('Error:', error));
}

window.onload = function() {
    fetch('/get_folders')
        .then(response => response.json())
        .then(folders => {
            const select = document.getElementById('folderSelect');
            folders.forEach(folder => {
                const option = document.createElement('option');
                option.value = folder;
                option.textContent = folder;
                select.appendChild(option);
            });
        })
        .catch(error => console.error('Error:', error));
};

function createTileCheckboxes() {
    fetch('/get_tile_strs')
        .then(response => response.json())
        .then(tileMapping => {
            const container = document.getElementById('tileCheckboxes');
            container.innerHTML = ''; // Clear previous content
            // Iterate through the list
            tileMapping.forEach(tileName => {
                // Create a checkbox for the tile
                const checkbox = document.createElement('input');
                checkbox.type = 'checkbox';
                checkbox.id = `tile-${tileName}`;
                checkbox.name = `tile-checkbox`;

                if (tileName === 'empty') {
                    checkbox.checked = true; // Set the `empty` tile as the default selected tile
                    selectedTileType = 'empty';
                }

                // Create an image element for the tile
                const img = document.createElement('img');
                img.src = `/static/tiles/${tileName}.png`; // Adjust the path as needed
                img.alt = tileName;
                img.style.margin = '0 10px'; // Add some spacing between the checkbox and the image

                // Create a label for better UX
                const label = document.createElement('label');
                label.htmlFor = `tile-${tileName}`;
                label.appendChild(document.createTextNode(tileName));

                // Append the checkbox, image, and label to the container
                container.appendChild(checkbox);
                container.appendChild(img);
                container.appendChild(label);
                container.appendChild(document.createElement('br')); // Line break for neatness
            });
        })
        .catch(error => console.error('Error:', error));
        // Set the `empty` tile as the default selected tile
}

function sendMap() {
    // Send the current display map back to the server 
    fetch('/update_map', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({map: currTileMap}),
    }).then(response => response.json())
}
// Call the function to create the checkboxes and images
createTileCheckboxes();


// Initialize the environment and start updating the tilemap periodically
initEnv(); // Call this function to initialize your environment, if necessary
// tickEnv();
// setInterval(tickEnv, 0); // Update the tilemap every 1000 milliseconds (1 second)
