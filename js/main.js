window.onload = async () => {
    /*
    const urlParams = new URLSearchParams(window.location.search).entries();
    const paramList  = {};
    for (pair of urlParams) {
        paramList[pair[0]] = pair[1];
    }
    */

    //set up the help menu - do this here to make it not
    //pop up for a brief second when the page loads in
    const helpWindow = document.querySelector(".help");
    const helpBackground = document.querySelector(".help-background");
    document.querySelector("#help").onclick = () => {
        helpWindow.style.display = "";
        helpBackground.style.display = "";
    };
    helpBackground.onclick = () => {
        helpWindow.style.display = "none";
        helpBackground.style.display = "none";
    };
    helpBackground.onclick();

    //compile all shaders
    const vox = await voxels(document.querySelector("canvas"), 512, {profiling: false});

    //random shit - remove later
    vox.setFOV(Math.PI / 4.);

    const centerposition = [0., 0., 0.];

    //---------------- Set up the UI ----------------//
    const flags = {"changed": true};

    //write current GPU device to screen, if available
    if (vox.adapterInfo) {
        const str = vox.adapterInfo["description"];
        let outstr = ""; const max_size = 21;
        if (str) {
            for (var i = 0; i < max_size - 3; i++) {
                if (str[i]) outstr += str[i];
            }
            if (str.length > max_size) {
                outstr += "...";
            } else {
                if (str[max_size - 2]) outstr += str[max_size - 2];
                if (str[max_size - 1]) outstr += str[max_size - 1];
                if (str[max_size - 0]) outstr += str[max_size - 0];
            }
        } else {
            outstr = "no info found";
        }
        document.querySelector("#adapter-info").innerHTML = outstr;
    }
     
    //download the current render result
    document.querySelector("#save-image").onclick = () => {vox.downloadImage();};

    //bind all sliders to update the values displayed next to them
    uiBindSlider(
        document.querySelector("#focal-distance-slider"),
        focalMap,
        document.querySelector("#focal-distance-value"),
        4,
        flags
    );
    uiBindSlider(
        document.querySelector("#aperture-slider"),
        a => a,
        document.querySelector("#aperture-value"),
        4,
        flags
    );
    uiBindSlider(
        document.querySelector("#sun-brightness-slider"),
        a => a,
        document.querySelector("#sun-brightness-value"),
        4,
        flags
    );
    uiBindSlider(
        document.querySelector("#sky-brightness-slider"),
        a => a,
        document.querySelector("#sky-brightness-value"),
        4,
        flags
    );
    uiBindSlider(
        document.querySelector("#azimuth-slider"),
        a => a,
        document.querySelector("#azimuth-value"),
        4,
        flags
    );
    uiBindSlider(
        document.querySelector("#zenith-slider"),
        a => a,
        document.querySelector("#zenith-value"),
        4,
        flags
    );
    
    //inputs that need updates but don't necessarily change anything else
    document.querySelector("#sky-color").oninput    = () => {flags["changed"] = true;};
    document.querySelector("#sun-color").oninput    = () => {flags["changed"] = true;};
    document.querySelector("#reset-render").onclick = () => {flags["changed"] = true;};

    //set up the tonemapping radio
    const tonemap = document.querySelector("#tone-mapping");
    const aces = document.querySelector("#aces");
    const reinhard = document.querySelector("#reinhard");
    const setACES = () => {
        aces.checked = true;
        reinhard.checked = false;
        tonemap.dataset.mode = 0;
    };
    const setReinhard = () => {
        aces.checked = false;
        reinhard.checked = true;
        tonemap.dataset.mode = 1;
    }
    aces.oninput = () => {
        console.log(tonemap.dataset.mode);
        if (tonemap.dataset.mode == 0) {return;}
        setACES();
        flags["changed"] = true;
    };
    reinhard.oninput = () => {
        if (tonemap.dataset.mode == 1) {return;}
        setReinhard();
        flags["changed"] = true;
    };
    setReinhard();
  
    //takes an array of buffers, parses them as .vox models,
    //then uploads a palette for each of the
    async function uploadModels(buffs) {
        const models = [];
        for (var x in buffs) {
            const curr = parseVox(buffs[x]);
            if (curr) {models.push(curr)}
        }

        if (models.length == 0) {alert("no available models!"); return;}

        vox.clearScene();

        let count = 0; let av = [0, 0, 0];

        //find the min/max bounds
        var mins = [1_000_000, 1_000_000, 1_000_000];
        for (var x in models) {
            setAll(models[x], (a, b, c, d) => {
                if (d == 0) {return;}
                mins[0] = Math.min(mins[0], a);
                mins[1] = Math.min(mins[1], b);
                mins[2] = Math.min(mins[2], c);

            });
        }

        for (var x in models) {
            setAll(models[x], (a, b, c, d) => {
                if (b > 0) {av[0] += a - mins[0]; av[1] += b - mins[1]; av[2] += c - mins[2]; count++;}
                vox.setVoxel(a - mins[0], b - mins[1], c - mins[2], d);
            });
        }

        centerposition[0] = av[0] / count;
        centerposition[1] = av[1] / count;
        centerposition[2] = av[2] / count;

        const palette = getPallete(models[0]);
        for (var x = 0; x < 256; x++) {
            vox.setMaterial(x, palette[x], [0., 0., 0., 0.], [[[1]]]);
        }

        vox.uploadMaterials();
        await vox.uploadScene();

        vox.setReset();
    }

    //functions to do with loading in .vox models
    document.querySelector("#upload-model").onclick = () => {
        const input = document.createElement("input"); input.type = "file";

        input.setAttribute("accept", ".vox");
        
        input.addEventListener("change", () => {
            const reader = new FileReader();
            const buffs  = []; let incr = 0;

            reader.onload = async () => {
                buffs.push(reader.result); incr++;
                if (incr < input.files.length) reader.readAsArrayBuffer(input.files[incr]);
                else await uploadModels(buffs);
            }
            reader.readAsArrayBuffer(input.files[incr]);
        }, {once: true});
        input.click();
    };

    //functions to allow users to drag and drop .vox files
    //from: https://developer.mozilla.org/en-US/docs/Web/API/HTML_Drag_and_Drop_API/File_drag_and_drop
    document.body.ondrop = (e) => {
        console.log("files dropped");

        e.preventDefault();

        //get all files dropped in
        const files = [];
        if (e.dataTransfer.items) {
            [...e.dataTransfer.items].forEach((item, i) => {
                if (item.kind === "file") {
                    const file = item.getAsFile();
                    files.push(file);
                }
            });
        } else {
            [...eval.dataTransfer.files].forEach((file, i) => {
                files.push(file);
            });
        }

        //read all files as buffers
        const reader = new FileReader();
        const buffs  = []; let incr = 0;

        reader.onload = async () => {
            buffs.push(reader.result); incr++;
            if (incr < files.length) reader.readAsArrayBuffer(files[incr]);
            else await uploadModels(buffs);
        }
        reader.readAsArrayBuffer(files[incr]);
    };

    document.body.ondragover = (e) => {e.preventDefault()}

    //movement / interaction with scene
    let theta = 5. * Math.PI / 4.;
    let phi = Math.PI / 8.;

    let deltaX = 0.; let deltaY = 0.;
    document.querySelector("#render-target").addEventListener("mousedown", (e) => {
        const mouse = {x: e.clientX, y: e.clientY};
        
        mousevelocity = {x: 0., y: 0.};
        function mouseMove(e) {
            deltaX += e.clientX - mouse.x; deltaY += e.clientY - mouse.y;
            mouse.x = e.clientX; mouse.y = e.clientY;
        }
        document.querySelector("#render-target").addEventListener("mousemove", mouseMove);
        document.body.addEventListener("mouseup", () => {
            document.querySelector("#render-target").removeEventListener("mousemove", mouseMove);
        }, {once : true});
    });

    let then = Date.now() * .001;
    async function frame() {
        let now = Date.now() * .001;
        const delta = now - then;
        then = now;

        if (deltaX != 0 || deltaY != 0) {vox.setReset();};
        theta -= deltaX * .002;
        phi   += deltaY * .002;

        phi = Math.min(Math.max(phi, -Math.PI * .499), Math.PI * .499);

        deltaX = 0.; deltaY = 0.;

        if (flags["changed"]) {
            vox.uploadRenderSettings(getUIValues());
        } flags["changed"] = false;

        const dist = 300;
        vox.setLookAt(centerposition);
        vox.setPosition([
            centerposition[0] + Math.cos(theta) * Math.cos(phi) * dist,
            centerposition[1] + Math.sin(theta) * Math.cos(phi) * dist,
            centerposition[2] + Math.sin(phi) * dist
        ]);

        let a = await vox.frame();

        if ("profiling" in a) {} 
        else {
            await a["done"];
        }

        document.querySelector("#samples-count").innerHTML = `${a["samples"]}`;
        window.requestAnimationFrame(frame);
    }
    frame();
};

//used to map linear slider value to exponential focal distance
function focalMap(a) {
    return Math.exp((a - .2) * 7.);
}

//writes all UI values to a typed array to be sent to the GPU
function getUIValues() {
    const returned = new Float32Array(4 + 4 + 4 + 4);
    const suncol = hexToRGB(document.querySelector("#sun-color").value);
    const skycol = hexToRGB(document.querySelector("#sky-color").value);
    returned[0] = focalMap(parseFloat(document.querySelector("#focal-distance-slider").value));
    returned[1] = parseFloat(document.querySelector("#aperture-slider").value);
    returned[2] = 3.1415;
    returned[3] = 3.1415;
    returned[4] = suncol[0]; returned[5] = suncol[1]; returned[6] = suncol[2];
    returned[7] = parseFloat(document.querySelector("#sun-brightness-slider").value);
    returned[8] = skycol[0]; returned[9] = skycol[1]; returned[10] = skycol[2];
    returned[11]= parseFloat(document.querySelector("#sky-brightness-slider").value);
    returned[12]=parseFloat(document.querySelector("#azimuth-slider").value * Math.PI / 180.);
    returned[13]=parseFloat(Math.PI / 2. + document.querySelector("#zenith-slider").value * Math.PI / 180.);
    returned[14]=document.querySelector("#tone-mapping").dataset.mode;
    return returned;
}

function hexToRGB(hex) {
    return [
        parseInt(hex[1] + hex[2], 16) / 255.,
        parseInt(hex[3] + hex[4], 16) / 255.,
        parseInt(hex[5] + hex[6], 16) / 255.
    ]
}

function uiBindSlider(slider, sliderCurve, text, maxChars, flags) {
    slider.oninput = () => {
        flags["changed"] = true;
        const val = sliderCurve(slider.value);
        let strVal = "" + val;
        let outstr = "";
        for (var x = 0; x < maxChars && x < strVal.length; x++) {
            outstr += strVal[x];
        }
        if (outstr.length < maxChars) {
            if (Math.floor(val) == val) {
                outstr += ".";
            }
            while(outstr.length < maxChars) {
                outstr += "0";
            }
        }
        text.innerHTML = outstr;
    }
    slider.oninput();
}