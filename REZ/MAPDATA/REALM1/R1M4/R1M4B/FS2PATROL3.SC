Name "fs2patrol3"

AbortOnDamage			FALSE		
AbortOnEnemySighted		FALSE		
SightRange			1200.0		
TimeOut				999.0
CheckForEnemyInterval		0.5		
TargetPlayer					
YawRate				360.0
CanBeMoved			FALSE		


Loop1:
OnEnemyOutOfRange_Goto		DISABLE		
OnTimerExpired_Goto		DISABLE	
OnEnemySighted_Goto		Attack		
OnDamage_Goto			FarAttack		
TimeOut				10.0


Walk						
StartAnim				"Walk"
TurnTo				target1
MoveTo				target1
TurnToTarget			0.57		
StartAnim	"AmbA"
Wait		1.0

//	turn to look at path2 ...
StartAnim	"Walk"
TurnTo		target2

//	then walk to path 2
StartAnim	"Walk"
MoveTo		target2
TurnToTarget	0.57

//	wait here for a bit
StartAnim	"AmbA"
Wait		1.0

//	walk to path3
Walk
YawRate		180
StartAnim	"Walk"
TurnTo		target3
MoveTo		target3

//	turn to face off of ledge
TurnToTarget	0.57
StartAnim	"AmbA"
Wait		1.0

//	walk back to path2 and start all over again
StartAnim	"Walk"
TurnTo		target2
MoveTo		target2

//	turn an look off of the ledge again
TurnToTarget	0.57
StartAnim	"AmbA"
Wait		1.0

Goto	Loop1

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
If_HealthLessThanQuarter_Goto	Crazy
if_RandomLessThan25_Goto	Attack_path1
if_RandomLessThan25_Goto	Attack_path2
if_RandomLessThan25_Goto	Attack_path3

//
//	Attack 2
//
Attack_path2:
TurnTo		target1
Run
StartAnim	"Run"

//	start a random timer that will make us stop sometime between when we start
//	running and when we reach the path...
OnTimerExpired_Goto		Attack_path2a
StartRandomTimer	0.5		2.0

MoveTo		target1

Attack_path2a:
TurnToTarget	0.33	//	this is the time start of attack to firing in anim
RangedAttackTarget
MeleeAttackTarget

If_HealthLessThanQuarter_Goto	Crazy
if_RandomLessThan50_Goto	Attack_path3

//
//	Attack 1
//
Attack_path1:
TurnTo		target2
Run
StartAnim	"Run"

//	start a random timer that will make us stop sometime between when we start
//	running and when we reach the path...
OnTimerExpired_Goto		Attack_path1a
StartRandomTimer	0.5		2.0

MoveTo		target2

Attack_path1a:
TurnToTarget	0.33	//	this is the time start of attack to firing in anim
RangedAttackTarget

If_HealthLessThanQuarter_Goto	Crazy
if_RandomLessThan50_Goto	Attack_path2

//
//	Attack 3
//
Attack_path3:
TurnTo		target3
Run
StartAnim	"Run"

//	start a random timer that will make us stop sometime between when we start
//	running and when we reach the path...
OnTimerExpired_Goto		Attack_path3a
StartRandomTimer	0.5		2.0

MoveTo		target3

Attack_path3a:
TurnToTarget	0.33	//	this is the time start of attack to firing in anim
RangedAttackTarget

If_HealthLessThanQuarter_Goto	Crazy
if_RandomLessThan50_Goto	Attack_path1

Goto	Attack_path2

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
		
	Goto				Crazy

