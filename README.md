<h1>rKPC</h1>

>An full remake of the game [Kiss: Psycho Circus: The Nightmare Child](https://en.wikipedia.org/wiki/Kiss:_Psycho_Circus:_The_Nightmare_Child) using the programming language Rust and the graphical API Vulkan.

<h2>How it came to be</h2>

It started as a [DLL injection project/D3D9 hook](http://https://github.com/scutyok/KPCmemoryHook "DLL injection project/D3D9 hook") but it has transpired into being a full KPC:TNC remake project. The original Lithtech 1.5 engine had it's physics FPS based, which led to speedruns of this game being very frustrating, locking to 24FPS being mandatory. This remake is made to address these issues as well to preserve the game in a modern, playable state.

https://github.com/user-attachments/assets/413c1559-ead3-4899-ad97-ca549fca8e11

<h2>Engine Capabilities</h2>

- Reading .DAT world files
	-	BVH based collisions
	-	Reading and rendering the BSP tree
	-	Reading objects
	-	Reading and applying world proprieties
- Reading, rendering and assigning .ABC models to objects
- Reading, rendering and assigning .DTX textures to surfaces and models
- Dynamic Lighting
- UV Lighting 
- Simple Skyboxes (six-sided)
- Complex Skyboxes (mirroring of models)
- Occlusion culling
- Lithtech 1.0 Fog

<h2>Roadmap</h2>

- Check TODO file

<h2>How to use (two ways)</h2>

<ol>
	<li><h3>Via Terminal</h3>
  with "cargo run --release"</li>
	<li><h3>Via Bat File</h3>
  open the "run_release.bat" file</li>
</ol>

<h2>Keybinds</h2>

| Key           | Function                      |
| ------------- | ----------------------------- |
| W             | Move forward                  |
| S             | Move backwards                |
| A             | Strafe left                   |
| D             | Strafe right                  |
| Space         | Ascend/Jump                   |
| Shift         | Descend                       |
| F1            | Open Debug menu               |
| F2            | Show triggers volume nodes    |
| ESC           | Unlock the cursor             |
| E             | Interact                      |

<h2>Thanks to</h2>

-	[Monolith for releasing their source code on various lithtech games which served as a guide](https://github.com/jsj2008/lithtech)
- 	[Sverre André Kvernmo (Cranium) for giving us the 99' design documents](https://doomwiki.org/wiki/Sverre_Andr%C3%A9_Kvernmo_(Cranium))
