const{ app, BrowserWindow } = require("electron")
require('electron-reload')(__dirname)

function createwindow(){
    const mainwindow = new BrowserWindow({
        width : 800,
        height : 600,
        autoHideMenuBar: true,
    })
    mainwindow.loadFile('./Homepage_2/Home.html')
}

app.whenReady().then(()=>{
    createwindow()
})