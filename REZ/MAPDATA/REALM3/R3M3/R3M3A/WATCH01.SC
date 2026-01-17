Name "watch01"

///////////////////////////////////////////////////////////////////////////////
//	set up some starting states for this script
///////////////////////////////////////////////////////////////////////////////

AbortOnDamage			FALSE		//	abort the script if we're damaged
AbortOnEnemySighted		FALSE		//	Don't abort if we sight an enemy
SightRange			2048.0		//	can see 1024 units
TimeOut				999.0
CheckForEnemyInterval		0.5		//	check for enemies every 1/2 second
TargetPlayer					//	target the player
YawRate				360.0
CanBeMoved			FALSE		//  don't allow outside forces to push the monster

///////////////////////////////////////////////////////////////////////////////
//	Patrolling, not attacking...
///////////////////////////////////////////////////////////////////////////////

Loop1:
OnEnemyOutOfRange_Goto		DISABLE		//	don't branch on enemy out of range

						//	don't branch if the timer expires!! This can happen if the enemy goes out
						//	of range during the Attack routine and the time is still ticking!!
OnTimerExpired_Goto		DISABLE	

OnEnemySighted_Goto		Attack		//	if we see the player, jump to Attack
OnDamage_Goto			FarAttack		//	if we get hit, jump to FarAttack
TimeOut				10.0


Walk						//	walk to the path3 and wait
SightRange			1024.0	
StartAnim				"Walk"
TurnTo				ball08pos01
MoveTo				ball08pos01
StartAnim	"AmbA"
Wait		2.0

//	turn to look at path2 ...
SightRange			100.0	
StartAnim	"Walk"
TurnTo		ball08pos02

//	then walk to path 2
StartAnim	"Walk"
MoveTo		ball08pos02

//	wait here for a bit
StartAnim	"AmbA"
Wait		1.0

//	walk to path1
SightRange			100.0	
Walk
YawRate		180
StartAnim	"Walk"
TurnTo		ball08pos03
MoveTo		ball08pos03

StartAnim	"AmbA"
Wait		4.0

//	walk back to path2 and start all over again
StartAnim	"Walk"
SightRange			100.0	
TurnTo		ball08pos02
MoveTo		ball08pos02

StartAnim	"AmbA"
Wait		1.0

Goto	Loop1

///////////////////////////////////////////////////////////////////////////////
//	Special attack... this happens if the enemy is out of range but manages
//	to hurt us somehow.  In that case, we jump here and set our range really
//	high so that we don't keep thinking our enemy is out of range
///////////////////////////////////////////////////////////////////////////////

FarAttack:
SightRange	2048.0

///////////////////////////////////////////////////////////////////////////////
//	do the meanie, attacking stuff here
///////////////////////////////////////////////////////////////////////////////

Attack:
OnDamage_Goto			DISABLE	//	don't branch if damaged while attacking
OnEnemySighted_Goto		DISABLE	//	don't branch if enemy sighted while attacking
OnEnemyOutOfRange_Goto		Loop1

YawRate		540
TimeOut		2.5

//	choose a random place to start from
If_HealthLessThanHalf_Goto	Crazy

TurnTo		ball08pos01
Run
StartAnim	"Run"

MoveTo		ball08pos01

Attack_path2a:
TurnToTarget	0.33	//	this is the time start of attack to firing in anim
RangedAttackTarget
MeleeAttackTarget

If_HealthLessThanHalf_Goto	Crazy

Goto	Attack

///////////////////////////////////////////////////////////////////////////////
//	When ballbuster is almost dead he goes Crazy.
///////////////////////////////////////////////////////////////////////////////
Crazy:
	LoopAnim	FALSE

	OnDamage_Goto			DISABLE
	OnEnemySighted_Goto		DISABLE
	OnEnemyOutOfRange_Goto		loop1
	
	TurnToTarget			0.57	
	RangedAttackTarget
	MeleeAttackTarget
		
	Goto				Crazy

