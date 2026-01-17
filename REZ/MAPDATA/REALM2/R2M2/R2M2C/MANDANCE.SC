Name "mandance"

///////////////////////////////////////////////////////////////////////////////
//	initial set up 
///////////////////////////////////////////////////////////////////////////////
AbortOnDamage			FALSE		
AbortOnEnemySighted		FALSE	
IgnoreAlertNear			TRUE
AlertGroup			TRUE
IgnoreAlertGroup			TRUE
SightRange			1		//  can see 
TimeOut				999.0
//WalkSpeed			70
//RunSpeed			120
YawRate				360.0
TargetPlayer					//  target the player
CanBeMoved			FALSE		//  don't allow outside forces to push the monster



///////////////////////////////////////////////////////////////////////////////
//	If he can see the player then he shoots at the player.
///////////////////////////////////////////////////////////////////////////////
Loop1:
	TurnToAngle 90
	LoopAnim	False
	StartAnim	"AmbA"
	Wait		1
	StartAnim "AmbB"
	Wait		1
	StartAnim "AmbA"
	Wait		1
	StartAnim "AmbB"
	Wait		1
	StartAnim "DeathB"	
	Wait		1
	StartAnim "AmbB"
	Wait		1
	StartAnim "Fall"
	Wait 		.5	

	TurnToAngle 180
	LoopAnim	False
	StartAnim	"AmbA"
	Wait		1
	StartAnim "AmbB"
	Wait		1
	StartAnim "AmbA"
	Wait		1
	StartAnim "AmbB"
	Wait		1
	StartAnim "DeathB"	
	Wait		1
	StartAnim "AmbB"	
	Wait		1
	StartAnim "Fall"
	Wait 		.5	

	TurnToAngle 270
	LoopAnim	False
	StartAnim	"AmbA"
	Wait		1
	StartAnim "AmbB"
	Wait		1
	StartAnim "AmbA"
	Wait		1
	StartAnim "AmbB"
	Wait		1
	StartAnim "DeathB"	
	Wait		1
	StartAnim "AmbB"
	Wait		1
	StartAnim "Fall"
	Wait 		.5	

	TurnToAngle	0
	LoopAnim	False
	StartAnim	"AmbA"
	Wait		1
	StartAnim "AmbB"
	Wait		1
	StartAnim "AmbA"
	Wait		1
	StartAnim "AmbB"
	Wait		1
	StartAnim "DeathB"	
	Wait		1	
	StartAnim "AmbB"
	Wait		1
	StartAnim "Fall"
	Wait 		.5	

Goto Loop1
