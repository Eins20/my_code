    var imagedir = "./raw_images";
    var edgedir = "./edges";
    var image_now = [];
    var edged_image = [];
    var fso = new ActiveXObject("Scripting.FileSystemObject");
    var fldr = fso.GetFolder(imagedir);
    var ff = new Enumerator(fldr.Files);
    var cnt1 = 0;
    for(;!ff.atEnd();ff.moveNext()){
        image_now.append(ff.item());
        cnt1 ++;
    }
    image_now = sorted(image_now);

    var fldr2 = fso.GetFolder(edgedir);
    var ff2 = new Enumerator(fldr2.Files);
    var cnt2 =0;
    for(;!ff2.atEnd();ff2.moveNext()) {
        print(ff2.item());
        edged_image.append(ff2.item());
        cnt2++;
    }
    edged_image = sorted(edged_image);

    if(cnt1!==cnt2) print("image numbers not equal!");

    var index = 0;

function image_change(index,cnt1,image_now,edged_image) {
    let myImage1 = document.getElementById("raw_image");
    let myImage2 = document.getElementById("edged_image");
    index++;
    index = index%cnt1;
    myImage1.setAttribute("src",image_now[index]);
    myImage2.setAttribute("src",edged_image[index]);
    return index;
}
setInterval(changeImg,400);//每隔400ms就换一个

