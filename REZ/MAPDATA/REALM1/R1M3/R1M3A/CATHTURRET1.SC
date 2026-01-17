Name "cathturret1"

AbortOnDamage			FALSE		
AbortOnEnemySighted		FALSE		
SightRange			1200.0		
TimeOut				999.0
CheckForEnemyInterval		0.5		
TargetPlayer					
YawRate				180.0
CanBeMoved			FALSE		


Loop1:
OnEnemyOutOfRange_Goto		DISABLE		
OnTimerExpired_Goto		DISABLE	
OnEnemySighted_Goto		Attack		
OnDamage_Goto			FarAttack		
TimeOut				10.0


Run						
StartAnim			"Run"
TurnTo				pathtarget1
MoveTo				pathtarget1
TurnToTarget			0.7		
StartAnim	"AmbA"
Wait		0.5

//	turn to look at path2 ...
StartAnim	"Run"
TurnTo		pathtarget2

//	then run to path 2
StartAnim	"Run"
MoveTo		pathtarget2
TurnToTarget	0.7

//	wait here for a bit
StartAnim	"AmbA"
Wait		0.5


//	turn to face off of ledge
TurnToTarget	0.7
StartAnim	"AmbA"
Wait		0.5

//	walk back to path2 and start all over again
StartAnim	"Run"
TurnTo		pathtarget2
MoveTo		pathtarget2

//	turn an look off of the ledge again
TurnToTarget	0.7
StartAnim	"AmbA"
Wait		0.5

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

YawRate		270
TimeOut		2.5

//	choose a random place to start from
if_RandomLessThan25_Goto	Attack_path1
if_RandomLessThan25_Goto	Attack_path2


//
//	Attack 2
//
Attack_path2:
TurnTo		pathtarget1
Run
StartAnim	"Run"

//	start a random timer that will make us stop sometime between when we start
//	running and when we reach the path...
OnTimerExpired_Goto		Attack_path2a
StartRandomTimer	0.5		1.5

MoveTo		pathtarget1

Attack_path2a:
TurnToTarget	0.7	//	this is the time start of attack to firing in anim
RangedAttackTarget
MeleeAttackTarget

if_RandomLessThan50_Goto	Attack_path1

//
//	Attack 1
//
Attack_path1:
TurnTo		pathtarget2
Run
StartAnim	"Run"

//	start a random timer that will make us stop sometime between when we start
//	running and when we reach the path...
OnTimerExpired_Goto		Attack_path1a
StartRandomTimer	0.5		1.5

MoveTo		pathtarget2

Attack_path1a:
TurnToTarget	0.7	//	this is the time start of attack to firing in anim
RangedAttackTarget
MeleeAttackTarget

If_HealthLessThanHalf_Goto	Stop
if_RandomLessThan50_Goto	Attack_path2


Goto	Attack_path2


Stop:
	
End

