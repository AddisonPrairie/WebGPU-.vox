//note: needs rewrite

//takes an array buffer and parses it as a set of .vox chunks
function parseVox(blob) {
    if (blob.length % 4 != 0) {
        const b2 = new ArrayBuffer(blob.byteLength + 4 - (blob.byteLength % 4));
        new Uint8Array(b2).set(new Uint8Array(blob));
        blob = b2;
    }
    const dv = new DataView(blob);
    if (
        String.fromCharCode(dv.getUint8(0, true)) + String.fromCharCode(dv.getUint8(1, true)) + String.fromCharCode(dv.getUint8(2, true))
        !== "VOX"
    ) {alert("loading file is not .vox!"); return  null;}

    //adds a chunk, its type, its range, and recursively
    //calls to do the same for all child chunks
    function getChunk(byteIndex) {
        const i32Index = byteIndex / 4;
        const id =  String.fromCharCode(dv.getUint8(byteIndex + 0, true)) + 
                    String.fromCharCode(dv.getUint8(byteIndex + 1, true)) + 
                    String.fromCharCode(dv.getUint8(byteIndex + 2, true)) + 
                    String.fromCharCode(dv.getUint8(byteIndex + 3, true));

        const contentBytes = dv.getInt32(byteIndex + 4, true);//i32[i32Index + 1];
        const childBytes   = dv.getInt32(byteIndex + 8, true);//i32[i32Index + 2];

        const newchunk = {
            byteStart: byteIndex,
            id: id,
            content: [byteIndex + 12, byteIndex + 12 + contentBytes],
            children: []
        };

        //if this chunk has no children
        if (childBytes == 0) {return newchunk;}

        //else scan through byte range and locate children
        let scanning = 0;
        while (scanning < childBytes) {
            const curByte= scanning + newchunk.content[1];
            //const curI32 = (curByte) / 4;

            newchunk.children.push(getChunk(curByte));
            //get the byte size of the content / children and offset by that
            let curByteSize = dv.getInt32(curByte + 4, true) + dv.getInt32(curByte + 8, true);//i32[curI32 + 1] + i32[curI32 + 2];

            scanning += curByteSize + 12;
        }
        
        return newchunk;
    }

    const root = getChunk(8);

    //get all the "material" nodes
    doAll(root, "MATL", (chunk) => {
        let ptr = chunk.content[0];

        chunk.material = {};
        chunk.material.id = dv.getInt32(ptr, true);
        chunk.material.properties = parseDict(ptr + 8, dv.getInt32(ptr += 4, true));

        if (chunk.material.properties.entries[0].val !== "_diffuse") {console.log(chunk.material)};
    });


    //get all of the "group" nodes
    doAll(root, "nGRP", (chunk) => {
        let ptr = chunk.content[0];
        chunk.group = {};

        chunk.group.nodeID = dv.getInt32(ptr, true);
        chunk.group.attr = parseDict(ptr + 8, dv.getInt32(ptr + 4, true));

        ptr = chunk.group.attr.byteEnd;

        chunk.group.numChildren = dv.getInt32(ptr, true);

        ptr += 4;

        chunk.group.children = [];
        for (var i = 0; i < chunk.group.numChildren; i++) {
            chunk.group.children.push({childID: dv.getInt32(ptr, true)});
            ptr += 4;
        }
    });

    //get all the "shape nodes"
    doAll(root, "nSHP", (chunk) => {
        let ptr = chunk.content[0];
        chunk.shape = {};

        chunk.shape.nodeID = dv.getInt32(ptr, true);
        chunk.shape.attr = parseDict(ptr + 8, dv.getInt32(ptr + 4));
        
        ptr = chunk.shape.attr.byteEnd;

        chunk.shape.numModels = dv.getInt32(ptr, true);
        ptr += 4;

        chunk.shape.models = [];
        for (var i = 0; i < chunk.shape.numModels; i++) {
            const newModel = {};
            newModel.modelID = dv.getInt32(ptr, true);
            ptr += 4;
            newModel.attr = parseDict(ptr + 4, dv.getInt32(ptr));
            ptr = newModel.attr.byteEnd;
            chunk.shape.models.push(newModel);
        }
    });

    //get all the "transform" nodes
    doAll(root, "nTRN", (chunk) => {
        const base32 = chunk.content[0] / 4;
        const  base8 = chunk.content[0];

        chunk.transform = {};

        chunk.transform.nodeID = dv.getInt32(base8, true);//i32[base32];

        const dict = parseDict(base8 + 8, dv.getInt32(base8 + 4, true));//[base32 + 1]);
        chunk.transform.attr = {dict: dict, num: dv.getInt32(base8 + 4, true)};

        const newbase8 = chunk.transform.attr.dict.byteEnd;
        const newbase32= chunk.transform.attr.dict.byteEnd / 4;

        chunk.transform.childID = dv.getInt32(newbase8, true);//i32[newbase32];
        chunk.transform.reserved= dv.getInt32(newbase8 + 4, true);//i32[newbase32 + 1]; //check that this should always be -1
        chunk.transform.layerID = dv.getInt32(newbase8 + 8, true);//i32[newbase32 + 2];

        chunk.transform.numFrames = dv.getInt32(newbase8 + 12, true);//i32[newbase32 + 3]; //check that this is > 0

        chunk.transform.frames = [];

        let ptr = newbase8 + 16;
        for (var x = 0; x < chunk.transform.numFrames; x++) {
            const newframe = {};

            newframe.attr = parseDict(ptr + 4, dv.getInt32(ptr, true));
            ptr = newframe.attr.byteEnd;
            for (var i = 0; i < newframe.attr.entries.length; i++) {
            if (newframe.attr.entries[i].key === "_t") {
                const str = newframe.attr.entries[i].val;
                const vals = str.split(" ");
                newframe.position = [parseInt(vals[0]), parseInt(vals[1]), parseInt(vals[2])];
            }
           }
            chunk.transform.frames.push(newframe);
        }
    });
    let iter = 0;
    //apply transforms to every model based off the information in the scene graph
    applyDown(1, [0., 0., 0.]);
    
    //descends scene graph with stored translation and applies to everything
    function applyDown(index, transform) {
        //find the actual chunk that this index is in
        let chunk = null;
        for (var x in root.children) {
            let checking = null;
            if (root.children[x].group) {checking = root.children[x].group;}
            if (root.children[x].transform) {checking = root.children[x].transform;}
            if (root.children[x].shape) {checking = root.children[x].shape;}
            if (checking) {
                if (checking.nodeID == index) {chunk = checking; break;}
            }
        }
        if (++iter > 60) {return;}
        if (chunk == null) {return;}
        let children = chunk.childID;
        if (children == null) {
            children = chunk.children;
            if (children == null) {
                const models = chunk.models;
                for (var x in models) {
                    root.children[models[x].modelID * 2 + 1].translate = [transform[0], transform[1], transform[2]];
                }
                return;
            }
            for (var x = 0; x < children.length; x++) {
                applyDown(children[x].childID, transform);
            }
        } else {
            let offset = [0., 0., 0.];
            if (chunk.frames.length > 0) {
                if ("position" in chunk.frames[0]) offset = [chunk.frames[0].position[0], chunk.frames[0].position[1], chunk.frames[0].position[2]];
            }
            applyDown(children, [transform[0] + offset[0], transform[1] + offset[1], transform[2] + offset[2]]);
        }
    }

    console.log(root);

    return {
        root: root,
        buffer: blob
    }

    //for specific types of chunks, get information for them
    function doAll(chunk, type, func) {
        if (chunk.id === type) {
            func(chunk);
        }
        if (chunk.children.length > 0) {
            for (var x in chunk.children) doAll(chunk.children[x], type, func);
        }
    }

    //parse a dictionary as specified here: https://github.com/ephtracy/voxel-model/blob/master/MagicaVoxel-file-format-vox-extension.txt
    function parseDict(byteStart, numEntries) {
        const returned = {entries: []};
        let ptr = byteStart;

        for (var x = 0; x < numEntries; x++) {
            let str0 = "";
            const size0 = dv.getInt32(ptr, true);
            ptr += 4;
            for (var j = 0; j < size0; j++) {
                str0 += String.fromCharCode(dv.getUint8(ptr + j, true));
            }
            ptr += size0;
            const size1 = dv.getInt32(ptr, true);
            let str1  = "";
            ptr += 4;
            for (var j = 0; j < size1; j++) {
                str1 += String.fromCharCode(dv.getUint8(ptr + j, true));
            }
            ptr += size1;
            returned.entries.push({key: str0, val: str1});
        }

        returned.byteStart = byteStart; returned.byteEnd = ptr;
        return returned;
    }
}

//takes a parsed set of .vox chunks and calls a callback to set
//every filled voxel w/ material in this set of chunks
function setAll(vox, callback) {
    setSub(vox.root, vox.buffer, callback, vox.translations, {val: 0});
}

//takes a set of .vox chunks w/ their buffer and calls a callback
//to set every filled voxel w/ material in this set of chunks
function setSub(chunk, buffer, callback, translations, incr) {
    if (chunk.children.length > 0) {
        for (var x in chunk.children) {
            setSub(chunk.children[x], buffer, callback, translations, incr);
        }
    }
    //if this is full of voxels
    if (chunk.id === "XYZI") {
        const u8 = new Uint8Array(buffer);
        const u32= new Uint32Array(buffer);

        let offset = chunk.translate;
        if (!offset) {offset = [0., 0., 0.];}

        const num = u32[chunk.byteStart / 4];
        for (var x = chunk.content[0]; x < chunk.content[1];) {
            callback(u8[x] + offset[0], u8[x + 1] + offset[1], u8[x + 2] + offset[2], u8[x + 3]);
            x += 4;
        }

        incr.val++;
    }
}

//get the palette as an array of rgb float colors
function getPallete(vox) {
    function getPaletteChunk(chunk) {
        if (chunk.id === "RGBA") {
            return chunk;
        }
        else {
            for (var x in chunk.children) {
                const check = getPaletteChunk(chunk.children[x]);
                if (check) {return check;}
            }
        }
        return null;
    }

    const paletteChunk = getPaletteChunk(vox.root);

    if (paletteChunk == null) alert("model does not have palette chunk!");
    else {
        const returned = [];
        const u8 = new Uint8Array(vox.buffer);

        for (var x = paletteChunk.content[0]; x < paletteChunk.content[1]; x += 4) {
            returned.push([u8[x] / 255., u8[x + 1] / 255., u8[x + 2] / 255.]);
        }

        return returned;
    }
    const returned = [];
    for (var i = 0; i < 256; i++) {
        returned.push([.1 + .9 * i / 256, .1 + .9 * i / 256, .1 + .9 * i / 256]);
    }
    returned[0] = [1., 1., 1.];
    returned[1] = [1., 0., 0.];
    
    return returned;
}